import numpy as np
from torch.utils.data import random_split
from transformers import BertTokenizer

# BERT tokenizer for parsing user input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', padding=True)

# max ids that can be created from user input
MAX_ID_LENGTH = 100

def encode(input):
    tokenized_inputs = tokenizer.encode_plus(
                            input, 
                            add_special_tokens=True, 
                            max_length=MAX_ID_LENGTH, 
                            padding='max_length', 
                            return_tensors="pt"
                        )
    return tokenized_inputs

def create_input_ids(input):
    return encode(input)['input_ids']

def create_attention_mask(input):
    return encode(input)['attention_mask']

def split_data(dataset):
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])
    return train_data, valid_data, test_data
