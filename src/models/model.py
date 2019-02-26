import pickle
from functools import partial

import torch

import numpy
from allennlp.common import Params
from allennlp.data import Vocabulary
from torch.nn import Module, Linear, Embedding, LeakyReLU
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.datasets import VisualQATrainDataset, my_collate
from src.utils.helpers import init_config, filter_config
from src.data_preprocessing.pretrained_embeddings import SavedEmbeddings


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
            saved_embs = SavedEmbeddings(pickle.load(f))

        self.embs = Embedding(self.vocab_size, embedding_dim=self.emb_size, padding_idx=0)
        emb_weights = numpy.zeros((self.vocab_size, self.emb_size), dtype=numpy.float32)
        for idx, word in tqdm(vocab.get_index_to_token_vocabulary("tokens").items()):
            emb_weights[idx] = saved_embs.get(word)
        self.embs.weight.data = torch.tensor(emb_weights)
        self.embs.weight.requires_grad = False
        # self.vectors = pickle.load(Path(emb_path).open("rb"))
        # self.vectors["UNK"] = numpy.zeros(question_emb_size)
        self.question_to_hidden = Linear(self.emb_size, self.hidden_size)
        self.image_to_hidden = Linear(self.image_emb_size, self.hidden_size)
        self.scores_layer = Linear(self.hidden_size, self.n_classes)
        self.lrelu = LeakyReLU()

    def forward(self, inputs):
        questions_idxs, image_emb = inputs
        question_embs = self.embs(questions_idxs)

        # mask = (questions_idxs != 0).type(torch.float32)
        # lengths = mask.sum(dim=1)
        # question_features = (question_embs * mask.unsqueeze(2)).sum(dim=1)
        # question_features /= lengths.unsqueeze(1)
        question_features = question_embs.sum(dim=1)

        question_features = self.lrelu(self.question_to_hidden(question_features))
        image_tensor = self.lrelu(self.image_to_hidden(image_emb))
        combined = question_features * image_tensor
        logits = self.scores_layer(combined)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    config = init_config()
    data_config = config.pop("data")
    data = VisualQATrainDataset(**filter_config(data_config, VisualQATrainDataset.__init__))
    dl = DataLoader(data, batch_size=12, collate_fn=partial(my_collate, vocab=data.vocab))
    x, y = next(iter(dl))
    model = BaselineModel(
        config=config["model"],
        vocab=data.vocab, embeddings_result_file=data_config["embeddings_result_file"])
    model(x)
