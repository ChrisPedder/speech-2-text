"""
Model to detect sarcam in newspaper headlines based on the kaggle sarcasm
detection dataset here: https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

Uses fasttext embeddings which can be downloaded from here: https://fasttext.cc/docs/en/english-vectors.html

"""
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from typing import List, Dict
from argparse import Namespace

from vectorizers import BertEmbeddingVectorizer, FasttextVectorizer

config = Namespace(
    data_path = "/data/data.csv",
    word_embedding_path = "/word_embeddings/",
    model_state_file = "model.pth",
    save_dir = "model_storage/sarcasm_detector",
    # Model hyperparams
    max_sequence_length = 50,
    hidden_dim = 300,
    bidirectional = False,
    learning_rate = 0.001,
    batch_size = 16,
    num_epochs = 5,
    early_stopping_criterion = 5,
)

class DataIterator:

    def __init__(self, batch_size, dataset_split):
        self.dataset_path = f'datasets/sarcasm_data/{dataset_split}.json'
        self.batch_size = batch_size

    def get_iter(self):
        for line in open(self.dataset_path, 'r'):
            json_line = json.loads(line)
            obj = {"article_link": json_line["article_link"],
                   "headline": json_line["headline"],
                   "is_sarcastic": json_line["is_sarcastic"]}
            yield obj

    def batched(self):
        _batch_size = getattr(self, 'batch_size')
        iterator = self.get_iter()
        while True:
            try:
                yield self._as_batch(
                    [next(iterator) for _ in range(_batch_size)]
                )
            except StopIteration:
                break

    def _as_batch(self, batch):
        links = []
        headlines = []
        is_sarcastic = []
        for obj in batch:
            links.append(obj['article_link'])
            headlines.append(obj['headline'])
            is_sarcastic.append(obj['is_sarcastic'])
        return {
            'links': links,
            'headlines': headlines,
            'is_sarcastic': is_sarcastic
        }


class SarcasmDetector(nn.Module):

    def __init__(self, config):
        super(SarcasmDetector, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vectorizer = FasttextVectorizer(
            max_sequence_length=self.config.max_sequence_length)
        self.embedding, self.embedding_size = \
            self.vectorizer.get_fasttext_embedding_layer()
        self.fc1 = nn.Linear(
            self.config.hidden_dim * (1 + self.config.bidirectional),
            2
        )

    def LSTM_layer(self, inputs):
        LSTM = nn.LSTM(self.embedding_size,
                       self.config.hidden_dim,
                       batch_first=True,
                       bidirectional=self.config.bidirectional)

        batch_size = inputs.size()[0]
        state_shape = batch_size, self.hidden_dim, self.embedding_size,
        outputs, (ht, ct) = LSTM(inputs)
        return ht[-1] if not self.config.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

    def forward(self, batch):
        seq_lengths = torch.LongTensor(list(map(len, batch)))
        seq_tensor = torch.Tensor(
            torch.zeros((len(batch), seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.Tensor(seq)

        embedding = self.embedding(seq_tensor)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedding, seq_lengths.cpu().numpy(), batch_first=True,
            enforce_sorted=False)

        output = self.LSTM_layer(embedding)
        output = self.fc1(output)
        return output

    def apply_to_batch(self, batch):
        vectorized_batch = self.vectorizer.vectorize_batch(
            batch["headlines"]
        )
        prediction = self(vectorized_batch)
        labels = F.one_hot(
            torch.from_numpy(
                np.array(
                    batch["is_sarcastic"]
                )
            ),
        num_classes=2
        ).float()
        return prediction, labels

    def train(self):
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.num_epochs):
            epoch_losses = []
            train_iter = DataIterator(
                dataset_split='train',
                batch_size=self.config.batch_size).batched()
            for batch in train_iter:
                optimizer.zero_grad()
                prediction, labels = self.apply_to_batch(batch)
                loss = loss_function(prediction, labels)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            print('train loss on epoch {} : {:.3f}'.format(
                epoch, np.mean(epoch_losses)))

            dev_losses = []
            dev_iter = DataIterator(dataset_split='dev',
                batch_size=self.config.batch_size).batched()

            for batch in dev_iter:
                with torch.no_grad():
                    optimizer.zero_grad()
                    prediction, labels = self.apply_to_batch(batch)
                    loss = loss_function(prediction, labels)

                    dev_losses.append(loss.item())
            print('dev loss on epoch {} : {:.3f}'.format(
                epoch, np.mean(dev_losses)))

        test_losses = []
        test_iter = DataIterator(dataset_split='test',
            batch_size=self.config.batch_size).batched()

        for batch in test_iter:
            with torch.no_grad():
                optimizer.zero_grad()
                prediction, labels = self.apply_to_batch(batch)
                loss = loss_function(prediction, labels)

                test_losses.append(loss.item())

        print('test loss: {:.3f}'.format(np.mean(test_losses)))

if __name__ == "__main__":
    model = SarcasmDetector(config)
    model.train()
