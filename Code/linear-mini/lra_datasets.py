# Code was adapted from https://github.com/guy-dar/lra-benchmarks
# Dar, G. (2023). lra-benchmarks. GitHub. https://github.com/guy-dar/lra-benchmarks.

import numpy as np
import pandas as pd
from functools import reduce
import torch
from glob import glob
from itertools import cycle

# Object for text benchmark (which uses IMDB dataset). Used in Google Colab (A100)
class ImdbDataset:
    def __init__(self, config, split='train'):       
        data_paths = {'train': "/content/drive/MyDrive/STAT 946 Project/datasets/aclImdb/train", 'eval': "/content/drive/MyDrive/STAT 946 Project/datasets/aclImdb/test"}
        split_path = data_paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"
        neg_inputs = zip(glob(neg_path+"/*.txt"), cycle([0]))
        pos_inputs = zip(glob(pos_path+"/*.txt"), cycle([1]))
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data[i]
        with open(data[0], 'r', encoding='utf-8') as fo:
            source = fo.read()
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = int(data[1])
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)

# Object for listops benchmark. Used in local experiments (2080 Ti)
class ListOpsDataset:
    def __init__(self, config, split='train'):
        
        data_paths = {'train': "/home/k4liang/946 project/hybrid-mini/datasets/lra_release/listops-1000/basic_train.tsv",
                      'eval': "/home/k4liang/946 project/hybrid-mini/datasets/lra_release/listops-1000/basic_val.tsv"}
        self.data = pd.read_csv(data_paths[split], delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)