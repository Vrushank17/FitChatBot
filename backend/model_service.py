from utils import create_attention_mask, create_input_ids
from transformers import BertForSequenceClassification
import torch
from data_filtering import get_tag_length, tags, intents
import random

def return_response(user_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=get_tag_length()).to(device)

    # load model parameters
    model.load_state_dict(torch.load('model.pth'))

    model.eval()

    X = create_input_ids(user_input)
    attention_mask = create_attention_mask(user_input)

    output = model(X, attention_mask=attention_mask)

    # get prediction tensor
    predicted = torch.argmax(output.logits, dim=1)
    tag = tags[predicted.item()]

    # get exact probabilities of each category and get highest probability
    probs = torch.softmax(output.logits, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >= 0.85:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return response
    else:
        return "Sorry, I don't understand what you saying"