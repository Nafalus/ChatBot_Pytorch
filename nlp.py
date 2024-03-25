import nltk
import numpy as np
# nltk.download('punkt')  DOWNLOAD DULU JANGAN LUPA
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized, words):
    tokenized = [stemming(w) for w in tokenized]

    bag = np.zeros(len(words), dtype=np.float32)
    for i, word, in enumerate(words):
        if word in tokenized:
            bag[i] = 1.0

    return bag