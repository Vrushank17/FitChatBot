import json
import torch
from utils import create_input_ids, create_attention_mask, split_data
from torch.utils.data import Dataset

with open('intents.json', 'r') as f:
    intents = json.load(f)

X_train = []
y_train = []
tags = []
attention_masks = []

tag_index = 0

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train, attention_masks):
        self.n_samples = len(X_train)
        self.X_train = X_train
        self.y_train = y_train
        self.attention_masks = attention_masks
    
    def __getitem__(self, index):
        X = self.X_train[index]
        y = self.y_train[index]
        attention_mask = self.attention_masks[index]

        return {
            'input_ids': X,
            'tag': y,
            'attention_mask': attention_mask,
        }

    def __len__(self):
        return self.n_samples

for intent in intents['intents']:
    tags.append(intent["tag"])
    for pattern in intent['patterns']:
        input_ids = create_input_ids(pattern)
        attention_masks.append(create_attention_mask(pattern))
        X_train.append(input_ids)
        y_train.append(tag_index)
    tag_index += 1

X_train = torch.cat(X_train, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
y_train = torch.tensor(y_train)

dataset = ChatDataset(X_train, y_train, attention_masks)

train_data, valid_data, test_data = split_data(dataset)

def get_data():
    return train_data, valid_data, test_data

def get_tag_length():
    return len(y_train)

