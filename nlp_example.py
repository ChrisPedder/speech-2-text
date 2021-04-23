import json
import gensim
from gensim.models import FastText

import torch.nn as nn
import torch.nn.functional as F

config = Namespace(
    data_path = "/data/data.csv",
    word_embedding_path = "/word_embeddings/",
    model_state_file = "model.pth",
    save_dir = "model_storage/sarcasm_detector",
    # Model hyperparams
    max_sequence_length = 50,
    input_dim=300,
    learning_rate=0.001,
    batch_size=16,
    num_epochs=25,
    early_stopping_criterion=5,
)

def parse_dataset():

    def parse_data(file):
        for l in open(file,'r'):
            yield json.loads(l)

    data = list(parse_data('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'))

    article_link= []
    headline = []
    is_sarcastic = []

    for item in data:
        article_link.append(item['article_link'])
        headline.append(item['headline'])
        is_sarcastic.append(item['is_sarcastic'])

def load_fastext():
    print('loading word embeddings...')
    embeddings_index = {}
    f = open('../input/fasttext/wiki.simple.vec',encoding='utf-8')
    for i, line in enumerate(f):
        values = line.strip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'found {i} word vectors')

    return embeddings_index

def get_fasttext_embedding_layer():
    # FloatTensor containing pretrained weights
    # Can use this to get embeddings of words by calling
    # "embedding(input)"
    # where input is the index of the word in the gensim glove dictionary

    embeddings_index=load_fastext()
    model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    weights = torch.FloatTensor(model.wv)
    embedding = nn.Embedding.from_pretrained(weights)
    return embedding
    embedding(input)


class SarcasmDetector(nn.Module):

    def __init__(self, config):
        super(SarcasmDetector, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim

    def forward(self, x_in, apply_softmax=False):
