import json
from nltk_utils import stem, bag_of_words, tokenize
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("intents.json", "r") as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words += w
        xy.append((w,tag))

ignored_words = ["?", ",", ".", "!"]
all_words = [stem(item) for item in all_words if item not in ignored_words]
all_words = sorted(set(all_words))

tags = sorted(set(tags))


X_train = []
Y_train = []
counter = 0
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)


X_train = (np.array(X_train))
Y_train = (np.array(Y_train))



class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    def __len__(self):
        return self.n_samples
    
#hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(Y_train)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimisers

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%100 == 0:
        print(f'Epoch [{(epoch+1)}/{num_epochs}],  loss: {loss.item():.4f}')
print(f'final loss,  loss: {loss.item():.4f}')


