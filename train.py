import json
from nltk_utils import stem, bag_of_words, tokenize
import numpy as np
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

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)
X_train = (np.array(X_train))
Y_train = (np.array(Y_train))

print(X_train)