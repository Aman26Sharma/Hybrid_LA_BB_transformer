# Code was adapted from https://github.com/guy-dar/lra-benchmarks
# Dar, G. (2023). lra-benchmarks. GitHub. https://github.com/guy-dar/lra-benchmarks.

from transformers import BertConfig
import torch
import ml_collections
from train_utils import create_learning_rate_scheduler

# Tokenizer for single characters
def make_char_tokenizer(allowed_chars, lowercase_input=False):
    # Need distinct characters for vocab
    allowed_chars = list(set(allowed_chars))

    def _tokenizer(x, max_length):
        # Tokenize vocabulary. Apply masking/padding
        x = x[:max_length]
        if lowercase_input:
            x = x.lower()
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_chars.index(c) + 1, x)) + ([0] * (max_length - n))
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_chars) + 1
    return _tokenizer

# Tokenizer for entire words (used in IMDB dataset)
def make_word_tokenizer(allowed_words, lowercase_input=False, allow_unk=True):
    # Need distinct characters for vocab
    allowed_words = list(set(allowed_words))
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def _tokenizer(x_str, max_length):
        # Tokenize vocabulary. Apply masking/padding
        if lowercase_input:
            x_str = x_str.lower()

        x = x_str.split()
        x = x[:max_length]
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_words.index(c) + 2 if c in allowed_words else UNK_TOKEN, x)) + \
                  ([PAD_TOKEN] * (max_length - n))
        if not allow_unk:
            assert UNK_TOKEN not in ids, "unknown words are not allowed by this tokenizer"
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_words) + 2
    return _tokenizer

##############################
# IMPORTANT FOR IMDB DATASET #
##############################
# Use UTF-8 encoded words for IMDB vocabulary plsu special characters present in dataset
utf8_tokenizer = make_char_tokenizer([chr(i) for i in range(10000)] + ['，', '、'])

# Listops Config
def get_listops_config():
    # Note: 96000 training samples
    TRAINING_SIZE = 96000
    VAL_SIZE = 2000
    AMT_EPOCHS = 20
        
    # Define config parameters for training loop
    config = ml_collections.ConfigDict()
    config.batch_size = 40
    config.eval_frequency = TRAINING_SIZE // config.batch_size
    config.total_eval_samples = VAL_SIZE
    config.total_train_samples = TRAINING_SIZE * AMT_EPOCHS
    config.learning_rate = 0.001
    config.weight_decay = 1e-5
    config.warmup_steps = int(config.eval_frequency * 0.1)
    config.tied_weights = False
    config.max_length = 2016
    config.tokenizer = make_word_tokenizer(list('0123456789') + ['[', ']', '(', ')', 'MIN', 'MAX', 'MEDIAN', 'SUM_MOD'])
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * rsqrt_decay", config)
    
    # Change some config paramters based on original BERT-mini model and task requirements
    model_config = BertConfig(
        # model_type = "bert",
        hidden_size = 256,
        hidden_act = "gelu",
        initializer_range = 0.02,
        vocab_size = config.tokenizer.vocab_size,
        hidden_dropout_prob = 0.1,
        num_attention_heads = 4,
        type_vocab_size = 2,
        max_position_embeddings = config.max_length,
        num_hidden_layers = 4,
        intermediate_size = 1024,
        attention_probs_dropout_prob = 0.1,
        num_labels = 10
    )
    
    return config, model_config

# IMDB/Text Config
def get_text_classification_config(num_labels=2):
    # Note: 25000 training samples
    TRAINING_SIZE = 25000
    VAL_SIZE = 2000
    AMT_EPOCHS = 10
    
    # Define config parameters for training loop
    config = ml_collections.ConfigDict()
    config.batch_size = 128
    config.eval_frequency = TRAINING_SIZE // config.batch_size
    config.total_train_samples = TRAINING_SIZE * AMT_EPOCHS
    config.total_eval_samples = -1
    config.learning_rate = 0.005
    config.weight_decay = 1e-5
    config.warmup_steps = int(config.eval_frequency * 0.1)
    config.tokenizer = utf8_tokenizer
    config.tied_weights = False
    config.max_length = 1024
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * rsqrt_decay", config)
    
    # Change some config paramters based on original BERT-mini model and task requirements
    model_config = BertConfig(
        model_type = "bert",
        hidden_size = 256,
        hidden_act = "gelu",
        initializer_range = 0.02,
        vocab_size = config.tokenizer.vocab_size,
        hidden_dropout_prob = 0.1,
        num_attention_heads = 4,
        type_vocab_size = 2,
        max_position_embeddings = config.max_length,
        num_hidden_layers = 4,
        intermediate_size = 1024,
        attention_probs_dropout_prob = 0.1,
        num_labels = num_labels
    )
    
    return config, model_config