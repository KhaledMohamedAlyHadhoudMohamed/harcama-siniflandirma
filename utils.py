import re
import numpy as np

def preprocess_text(text):
    text = text.upper()
    text = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", "", text)
    return text.strip().split()

def average_word2vec(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)