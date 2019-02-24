import logging
import pickle
from typing import Dict

import torch
from pathlib import Path

import numpy
from allennlp.common import Params
from allennlp.data import Vocabulary
from torch.nn import Module, Linear, Embedding
from tqdm import tqdm

from src.utils.datasets import VisualQADataset
from src.utils.helpers import init_config, filter_config
from src.utils.pretrained_embeddings import SavedEmbeddings


class BaselineModel(Module):
    def __init__(self,
                 embeddings_result_file,
                 vocab: Vocabulary,
                 config: Params):
        """
        Gets sentence embedding b averaging w2v word reprsentations and image embedding from pretrained
        convnet, combines them by a dot-product, then applies logistic regresssion
        """
        super().__init__()
        self.emb_size = config.pop("emb_size")
        self.vocab_size = vocab.get_vocab_size("tokens")
        self.hidden_size = config.pop("hidden_size")
        self.image_emb_size = config.pop("image_emb_size")
        self.n_classes = config.get("n_classes")

        with open(embeddings_result_file, "rb") as f:
            saved_embs: SavedEmbeddings = pickle.load(f)

        self.embs = Embedding(self.vocab_size, embedding_dim=self.emb_size, padding_idx=0)
        emb_weights = numpy.zeros((self.vocab_size, self.emb_size), dtype=numpy.float32)
        for idx, word in tqdm(vocab.get_index_to_token_vocabulary("tokens").items()):
            emb_weights[idx] = saved_embs.get(word)
        self.embs.weight.data = torch.tensor(emb_weights)
        # self.vectors = pickle.load(Path(emb_path).open("rb"))
        # self.vectors["UNK"] = numpy.zeros(question_emb_size)
        self.question_to_hidden = Linear(self.emb_size, self.hidden_size)
        self.image_to_hidden = Linear(self.image_emb_size, self.hidden_size)
        self.scores_layer = Linear(self.hidden_size, self.n_classes)

    def forward(self, inputs):
        questions_idxs, image_emb = inputs
        question_embs = self.embs(questions_idxs)
        question_features = question_embs.mean(dim=-1)
        question_features = self.question_to_hidden(question_features)
        image_tensor = self.image_to_hidden(image_emb)
        combined = question_features * image_tensor
        logits = self.scores_layer(combined)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    config = init_config()
    data_config = config.pop("data")
    data = VisualQADataset(**filter_config(data_config, VisualQADataset.__init__))
    model = BaselineModel(
        config=config["model"],
        vocab=data.vocab, embeddings_result_file=data_config["embeddings_result_file"])
