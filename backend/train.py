from data_filtering import get_data, get_tag_length

from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BertForSequenceClassification, AdamW

# get datasets from data_filtering
train_dataset, valid_dataset, test_dataset = get_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT Seq Classification model that will be finetuned
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=get_tag_length()).to(device)

# hyperparameters for model
batch_size = 8
learning_rate = 5e-5
epochs = 7

# data loaders used for training, validation, and testing
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_size = batch_size,
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
)

# define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for epoch in range(epochs):

    progress_bar = tqdm(total=len(train_loader), desc='Training Progress')

    avg_train_loss = 0

    # training loop
    model.train()
    for batch in train_loader:
        X = batch['input_ids'].to(device)
        y = batch['tag'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Compute prediction and loss
        output = model(input_ids=X, attention_mask=attention_mask)

        train_loss = loss_fn(output.logits, y)
        avg_train_loss += train_loss.item()

        # Backpropagation
        train_loss.backward()

        # Set the maximum gradient norm
        max_grad_norm = 1.0
        clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    f1_avg = 0
    avg_valid_loss = 0

    # validation loop
    model.eval()
    for batch in valid_loader:
        X = batch['input_ids'].to(device)
        y = batch['tag'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Compute prediction and loss without calculating gradients
        with torch.no_grad():
            output = model(input_ids=X, attention_mask=attention_mask)
        
        predicted = torch.argmax(output.logits, dim=1)
        
        # calcuate f1 score
        f1_metric = f1_score(y, predicted, average="macro")
        f1_avg += f1_metric

        valid_loss = loss_fn(output.logits, y)
        avg_valid_loss += valid_loss.item()
    
    progress_bar.close()

    avg_train_loss /= len(train_loader)
    avg_valid_loss /= len(valid_loader)
    f1_avg /= len(valid_loader)
    
    # evaluation metrics
    print(f"training loss: {avg_train_loss:.4f}  [{epoch+1}/{epochs}]")
    print(f"validation loss: {avg_valid_loss:.4f}  [{epoch+1}/{epochs}]")
    print(f"f1 score: {f1_avg:.4f}  [{epoch+1}/{epochs}]")
    print(f"perplexity: {avg_valid_loss ** 2:.4f}  [{epoch+1}/{epochs}]")

def evaluate_accuracy(model, test_dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            X = batch['input_ids'].to(device)
            y = batch['tag'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=X, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(predicted_labels == y).item()
            total_samples += len(y)

    accuracy = correct_predictions / total_samples
    return accuracy

test_accuracy = evaluate_accuracy(model, test_loader)
print("Test Accuracy:", test_accuracy)

# save model for later use
torch.save(model.state_dict(), 'model.pth')

