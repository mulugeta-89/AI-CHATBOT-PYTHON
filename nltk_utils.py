import nltk
#nltk.download("punkt")
import numpy as np
from nltk import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, item in enumerate(words):
        if item in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag


