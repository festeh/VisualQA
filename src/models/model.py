import pickle
from functools import partial

import torch

import numpy
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from torch.nn import (
    Module,
    Linear,
    Embedding,
    LeakyReLU,
    Dropout,
    Tanh,
    LSTM,
    BatchNorm1d,
    LayerNorm,
    GRU,
    Softmax,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.datasets import VisualQATrainDataset, my_collate
from src.utils.helpers import init_config, filter_config
from src.data_preprocessing.pretrained_embeddings import SavedEmbeddings

from torch.nn.utils.weight_norm import weight_norm, WeightNorm


class Attention(Module):
    def __init__(
            self,
            question_emb_size,
            n_image_features,
            image_emb_size,
            hidden_size,
            dropout_prob,
    ):
        super().__init__()

        self.q_to_hidden = Linear(question_emb_size, hidden_size)
        self.im_to_hidden = Linear(image_emb_size, hidden_size)

        self.qnorm = weight_norm(self.q_to_hidden, "weight", dim=None)

        self.imnorm = weight_norm(self.im_to_hidden, "weight", dim=None)

        self.lrelu = LeakyReLU()
        self.softmax = Softmax(dim=1)

        self.dropout = Dropout(dropout_prob)

    def forward(self, question_emb, image_emb):
        question_emb = self.qnorm(question_emb)
        question_emb = self.dropout(self.lrelu(question_emb))
        image_emb = self.imnorm(image_emb)
        image_emb = self.dropout(self.lrelu(image_emb))

        combined_emb = torch.bmm(image_emb, question_emb.unsqueeze(2)).squeeze(2)
        probs = self.softmax(combined_emb)
        return probs


class VQAModel(Module):
    def __init__(self, embeddings_result_file, vocab: Vocabulary, config: Params):
        """
        Implements attention-based model
        """
        super().__init__()
        self.emb_size = config.pop("emb_size")
        self.vocab_size = vocab.get_vocab_size("tokens")
        self.hidden_size = config.pop("hidden_size")
        self.image_emb_size = config.pop("image_emb_size")
        self.n_image_features = config.pop("n_image_features")
        self.n_classes = config.pop("n_classes")
        self.dropout_rate = config.pop("dropout_rate")

        with open(embeddings_result_file, "rb") as f:
            saved_embs = SavedEmbeddings(pickle.load(f))

        self.embs = Embedding(
            self.vocab_size, embedding_dim=self.emb_size, padding_idx=0
        )
        emb_weights = numpy.zeros((self.vocab_size, self.emb_size), dtype=numpy.float32)
        saved_embs.return_zero_for_oov = False
        for idx, word in tqdm(vocab.get_index_to_token_vocabulary("tokens").items()):
            if idx != 0:
                emb_weights[idx] = saved_embs.get(word)
        self.embs.weight.data = torch.tensor(emb_weights)

        self.seq_embedder = PytorchSeq2VecWrapper(
            LSTM(
                input_size=self.emb_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=False,
            )
        )

        self.attention_module = Attention(
            question_emb_size=self.hidden_size,
            n_image_features=self.n_image_features,
            image_emb_size=self.image_emb_size,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout_rate
        )

        self.image_to_hidden = Linear(self.image_emb_size, self.hidden_size)
        self.question_to_hidden = Linear(self.hidden_size, self.hidden_size)

        self.hidden_to_hidden = Linear(self.hidden_size, self.hidden_size)

        self.scores_layer = Linear(self.hidden_size, self.n_classes)
        self.lrelu = LeakyReLU()

        self.lnimg = LayerNorm(self.image_emb_size)
        self.lnq = LayerNorm(self.hidden_size)

        self.dropout = Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        questions_idxs, image_embs = inputs

        question_embs = self.embs(questions_idxs)
        mask = (questions_idxs != 0).type(torch.float32)
        question_features = self.seq_embedder(question_embs, mask)
        att_probs = self.attention_module(question_features, image_embs)
        image_embs = torch.bmm(att_probs.unsqueeze(1), image_embs).squeeze(1)

        question_features = self.lnq(question_features)
        question_features = self.lrelu(self.question_to_hidden(question_features))

        image_emb = self.lnimg(image_embs)
        image_emb = self.lrelu(self.image_to_hidden(image_emb))

        combined = question_features * image_emb
        combined = self.dropout(combined)

        combined = self.lrelu(self.hidden_to_hidden(combined))
        combined = self.dropout(combined)

        logits = self.scores_layer(combined)
        return logits

    #
    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    config = init_config()
    data_config = config.pop("data")
    data = VisualQATrainDataset(
        **filter_config(data_config, VisualQATrainDataset.__init__)
    )
    dl = DataLoader(
        data, batch_size=12, collate_fn=partial(my_collate, vocab=data.vocab)
    )
    x, y = next(iter(dl))
    model = VQAModel(
        config=config["model"],
        vocab=data.vocab,
        embeddings_result_file=data_config["embeddings_result_file"],
    )
    model(x)
