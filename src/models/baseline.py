import pickle
import torch
from pathlib import Path

import numpy
from torch.nn import Module, Linear


class BaselineModel(Module):
    def __init__(self, question_emb_size, image_emb_size, n_classes, h_size, emb_path):
        """
        Gets sentence embedding b averaging w2v word reprsentations and image embedding from pretrained
        convnet, combines them by a dot-product, then applies logistic regresssion
        """
        super().__init__()
        self.vectors = pickle.load(Path(emb_path).open("rb"))
        self.vectors["UNK"] = numpy.zeros(question_emb_size)
        self.question_to_hidden = Linear(question_emb_size, h_size)
        self.image_to_hidden = Linear(image_emb_size, h_size)
        self.scores_layer = Linear(h_size, n_classes)

    def forward(self, questions: str, image_emb):
        q_embs = [[self.vectors[w] for w in q.split() if w in self.vectors] for q in questions]
        question_tensor = [torch.tensor(numpy.mean(q, axis=0), device=self.device) for q in q_embs]
        question_tensor = torch.stack(question_tensor)
        question_tensor = self.question_to_hidden(question_tensor)
        image_tensor = self.image_to_hidden(image_emb)
        combined = question_tensor * image_tensor
        logits = self.scores_layer(combined)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device
