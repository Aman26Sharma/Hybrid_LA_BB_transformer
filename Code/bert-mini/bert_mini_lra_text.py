# Code was adapted from https://github.com/guy-dar/lra-benchmarks
# Dar, G. (2023). lra-benchmarks. GitHub. https://github.com/guy-dar/lra-benchmarks.

from transformers import BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter

import json
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from lra_config import (get_listops_config, get_text_classification_config)
from lra_datasets import (ListOpsDataset, ImdbDataset)
from argparse import ArgumentParser

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
    
    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False  
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  
            return False

# Push inputs to CUDA device
def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

# How to process batches
def transformers_collator(sample_list):
    input_list, target_list = zip(*sample_list)
    keys = input_list[0].keys()
    inputs = {k: torch.cat([inp[k] for inp in input_list], dim=0) for k in keys}
    target = torch.cat(target_list, dim=0)
    return inputs, target

# Compute accuracy
def accuracy_score(outp, target):
    assert len(outp.shape) == 2, "accuracy score must receive 2d output tensor"
    assert len(target.shape) == 1, "accuracy score must receive 1d target tensor"
    return (torch.argmax(outp, dim=-1) == target).sum().item() / len(target)

# Constants
OUTPUT_DIR = "output_dir/"
deepspeed_json = "ds_config.json"

# Tasks
TASKS = {
         'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
         'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
        }

# Retrieve model
def get_model(config, model_config):
    # Print model configs (sanity check)
    print(model_config)
    
    # Create model
    model = BertForSequenceClassification(model_config)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
        
    # Print model architecture
    print(model)
    return model

# Training function
def train(model, config, use_deepspeed, early_stopper):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    lr = config.learning_rate
    wd = config.weight_decay
    batch_size = config.batch_size
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    avg_factor = 0.95
    
    # Create a SummaryWriter to save logs
    writer = SummaryWriter("/content/drive/MyDrive/STAT 946 Project/bert-mini/runs/text")
    
    dataset = task.dataset_fn(config, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=transformers_collator)
    eval_dataset = task.dataset_fn(config, split='eval')    
    max_train_steps = int(np.ceil(config.total_train_samples / batch_size))
    if config.total_eval_samples < 0:
        max_eval_steps = len(eval_dataset) // batch_size
    else:
        max_eval_steps = int(np.ceil(4 * config.total_eval_samples / batch_size))
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler_fn = config.lr_scheduler
    scheduler = scheduler_fn(optimizer)
    
    if use_deepspeed:
        with open(deepspeed_json, "r") as fp:
            deepspeed_config = json.load(fp)
        model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                                             optimizer=optimizer, lr_scheduler=scheduler,
                                                             config_params=deepspeed_config)
    # train model
    model.to(device)
    model.train()
    avg_loss = None
    avg_acc = None
    pbar = tqdm(cycle(dataloader), total=max_train_steps)
        
    # Print important information
    print(f"Total training dataset size: {len(dataset)}")
    print(f"Total evaluation dataset size: {len(eval_dataset)}")
    print(f"Length of Dataloader: {len(dataloader)}")
    print(f"Size of each batch: {dataloader.batch_size}")
    print(f"Total training steps: {config.total_train_samples}")
    print(f"Maximum number of training steps: {max_train_steps}")
    
    # Evaluation parameters for early stop
    eval_counter = 0
    
    for i, (inputs, target) in enumerate(pbar):
        if i == max_train_steps:
            break
        if use_deepspeed:
            outputs = model_engine(**inputs)
            loss = F.cross_entropy(outputs.logits, target)
            model_engine.backward(loss)
            model_engine.step()
        else:
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                
            inputs = dict_to_device(inputs, device)
            target = target.to(device)
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, target)
            loss.backward()
            if (i+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()

        cur_loss = loss.item()
        cur_acc = accuracy_score(outputs.logits, target)
        avg_loss = cur_loss if avg_loss is None else avg_factor * avg_loss + (1-avg_factor) * cur_loss  
        avg_acc = cur_acc if avg_acc is None else avg_factor * avg_acc + (1-avg_factor) * cur_acc
        pbar.set_postfix_str(f"loss: {avg_loss:.5f} accuracy: {avg_acc:.5f}")
        
        writer.add_scalar("Loss/train", cur_loss, i)
        writer.add_scalar("Accuracy/train", cur_acc, i)
        
        # evaluate
        if (config.eval_frequency > 0) and ((i+1) % config.eval_frequency == 0):
            eval_counter += 1
            model.eval()
            eval_running_loss = 0.
            eval_running_acc = 0.
            average_eval_loss = 0.
            average_eval_acc = 0.
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size // 4, 
                                         collate_fn=transformers_collator)
            eval_pbar = tqdm(eval_dataloader, total=max_eval_steps)
                        
            for j, (inputs, target) in enumerate(eval_pbar):
                if j == max_eval_steps:
                    break
                inputs = dict_to_device(inputs, device)
                target = target.to(device)
                outputs = model(**inputs)
                loss = F.cross_entropy(outputs.logits, target)
                eval_running_loss += loss.item()
                eval_running_acc += accuracy_score(outputs.logits, target)
                average_eval_loss = eval_running_loss / (j+1)
                average_eval_acc = eval_running_acc / (j+1)
                
                eval_pbar.set_postfix_str(f"eval loss: {average_eval_loss:.5f} "
                                          f"eval accuracy: {average_eval_acc:.5f}")
                
            writer.add_scalar("Loss/eval", average_eval_loss, eval_counter)
            writer.add_scalar("Accuracy/eval", average_eval_acc, eval_counter)
                              
            # Early Stopping
            if early_stopper(average_eval_acc):
                break
                    
            # Continue training
            model.train()
            
# main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(),
                        help="choose an LRA dataset from available options")
    parser.add_argument("--deepspeed", action="store_true",
                        help="use deepspeed optimization for better performance")
    args = parser.parse_args()
    task_name = args.task
    if args.deepspeed:
        import deepspeed
    
    task = TASKS[task_name]
    config, model_config = task.config_getter()    
    model = get_model(config, model_config)
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)
    train(model, config, use_deepspeed=args.deepspeed, early_stopper=early_stopper)
