import json
import torch
from utils import create_input_ids, create_attention_mask, split_data
from torch.utils.data import Dataset

# open intents.json file, which contains data
with open('intents.json', 'r') as f:
    intents = json.load(f)

# X_data stores encoded patterns
# y_data stores tag indices
# attention masks are needed for BERT

X_data = []
y_data = []
attention_masks = []

# contains actual tags as words
tags = []

# used to assign numerical labels to tags or topics
tag_index = 0

# PyTorch custom dataset
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data, attention_masks):
        self.n_samples = len(X_data)
        self.X_data = X_data
        self.y_data = y_data
        self.attention_masks = attention_masks
    
    def __getitem__(self, index):
        X = self.X_data[index]
        y = self.y_data[index]
        attention_mask = self.attention_masks[index]

        return {
            'input_ids': X,
            'tag': y,
            'attention_mask': attention_mask,
        }

    def __len__(self):
        return self.n_samples

# parsing the json file
for intent in intents['intents']:
    tags.append(intent["tag"])

    for pattern in intent['patterns']:
        #  uses utils file to parse the pattern
        input_ids = create_input_ids(pattern)
        attention_masks.append(create_attention_mask(pattern))

        # add data
        X_data.append(input_ids)
        y_data.append(tag_index)

    tag_index += 1

# convert necessary info into tensors
X_data = torch.cat(X_data, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
y_data = torch.tensor(y_data)

# create dataset
dataset = ChatDataset(X_data, y_data, attention_masks)

# split dataset into train, validation, and test
train_data, valid_data, test_data = split_data(dataset)

def get_data():
    return train_data, valid_data, test_data

def get_tag_length():
    return len(y_data)

