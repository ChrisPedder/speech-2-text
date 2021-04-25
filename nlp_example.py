"""
Model to detect sarcam in newspaper headlines based on the kaggle sarcasm
detection dataset here: https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

Uses fasttext embeddings which can be downloaded from here: https://fasttext.cc/docs/en/english-vectors.html

"""
import io
import json
import gensim
from gensim.models import FastText
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    "data_path": "/data/data.csv",
    "word_embedding_path": "/word_embeddings/",
    "model_state_file": "model.pth",
    "save_dir": "model_storage/sarcasm_detector",
    # Model hyperparams
    "max_sequence_length": 50,
    "input_dim": 300,
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 25,
    "early_stopping_criterion": 5,
}

def parse_dataset():

    def parse_data(file):
        for l in open(file,'r'):
            yield json.loads(l)

    data = list(parse_data('sarcasm_datasets/Sarcasm_Headlines_Dataset.json'))

    article_link= []
    headline = []
    is_sarcastic = []

    for item in data:
        article_link.append(item['article_link'])
        headline.append(item['headline'])
        is_sarcastic.append(item['is_sarcastic'])

    return zip(headline, is_sarcastic)

def load_fastext(fname):
    print('loading word embeddings...')
    embeddings_index = {}
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}

    for i, line in enumerate(fin):
        values = line.strip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    fin.close()
    print(f'found {i} word vectors')

    return embeddings_index


def get_fasttext_embedding_layer():
    # FloatTensor containing pretrained weights
    # Can use this to get embeddings of words by calling
    # "embedding(input)"
    # where input is the index of the word in the gensim glove dictionary

    embeddings_index=load_fastext("word_embeddings/wiki-news-300d-1M.vec")
    weights = torch.FloatTensor(model.wv)
    embedding = nn.Embedding.from_pretrained(weights)
    return embedding


class SarcasmDetector(nn.Module):

    def __init__(self, config):
        super(SarcasmDetector, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim

    def forward(self, x_in, apply_softmax=False):





if __name__ == "__main__":
    load_fastext("word_embeddings/wiki-news-300d-1M.vec")
