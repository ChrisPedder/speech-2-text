"""
Routines to vectorize text.

BertEmbeddingVectorizer uses BERT embeddings from the huggingface transformers
package.

FasttextVectorizer uses fasttext embeddings which can be downloaded from
here: https://fasttext.cc/docs/en/english-vectors.html and are stored in
the path ./word_embeddings

"""

import io
import re
import torch

import numpy as np
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from typing import List, Dict

# Set random seed for subsequent code for reproduceability
np.random.seed(42)

class BertEmbeddingVectorizer(object):

    def __init__(self, model_type='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_type)
        self.model = BertModel.from_pretrained(model_type)

    def vectorize_batch(self, sentences: List[str]):
        inputs = self.tokenizer(sentences,
                                return_tensors="pt",
                                padding='max_length',
                                max_length=128)
        outputs = self.model(**inputs,
                             output_hidden_states=True)

        last_hidden_states = outputs.hidden_states
        return last_hidden_states

class FasttextVectorizer(object):

    def __init__(self, max_sequence_length=128,
        file_name="word_embeddings/wiki-news-300d-1M.vec"):
        self.file_name = file_name
        self.embedding_dict, self.token_dict = self.load_fastext(file_name)
        self.max_sequence_length = max_sequence_length

    def load_fastext(self, file_name):
        print('loading word embeddings...')
        fin = io.open(self.file_name, 'r', encoding='utf-8', newline='\n',
            errors='ignore')
        n, d = map(int, fin.readline().split())

        token_dict = {}
        embedding_dict = {}

        for i, line in enumerate(fin):
            values = line.strip().rsplit(' ')
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            token_dict[word] = i + 2
            embedding_dict[i + 2] = coeffs
            # val 0 == <PAD>
            # val 1 =0 <UNK>
        fin.close()
        print(f'found {i+1} word vectors')

        # Set entries for padding and unknown chars
        token_dict['<PAD>'] = 0
        token_dict['<UNK>'] = 1

        embedding_dict[0] = np.random.rand(300)
        embedding_dict[1] = np.random.rand(300)

        return embedding_dict, token_dict

    def pad_sequence(self, token_list: List[str]) -> List[str]:
        """
        Pads sequences which are shorter than self.max_sequence_length to
        self.max_sequence_length. Crops sentences which are longer to
        self.max_sequence_length
        """
        if len(token_list) < self.max_sequence_length:
            pad_seq = ['<PAD>'] * (self.max_sequence_length - len(token_list))
            token_list = token_list + pad_seq
        elif len(token_list) > self.max_sequence_length:
            token_list = token_list[:self.max_sequence_length]

        return token_list

    def clean_text(self, text: str) -> List[str]:
        alpha_re = re.compile('[^a-zA-Z0-9 ]')
        alpha_re.sub('', text)
        text = text.strip()
        return text.split(' ')

    def vectorize(self, text: str, pad_seqs=False) -> np.ndarray:
        text_list = self.clean_text(text)
        if pad_seqs:
            text_list = self.pad_sequence(text_list)
        vectorize_list = [self.token_dict.get(word, 1) for word in text_list]
        return np.array(vectorize_list)

    def vectorize_batch(self, sentences: List[str], pad_seqs=False) -> np.ndarray:
        vectorize_list = [self.vectorize(text) for text in sentences]
        if pad_seqs:
            return np.stack(vectorize_list, axis=0)
        return vectorize_list
        
    def get_fasttext_embedding_layer(self):
        # Can use this to get embeddings of words by calling
        # "embedding(input)"
        # where input is the index of the word in token dictionary
        weights = np.stack([val for val in self.embedding_dict.values()], axis=0)
        # FloatTensor containing pretrained weights
        tensor_weights = torch.FloatTensor(weights)
        embedding = nn.Embedding.from_pretrained(tensor_weights)
        return embedding, weights.shape[1]
