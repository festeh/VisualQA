import pickle
from typing import Dict

import numpy
from gensim.models import KeyedVectors

from src.utils.helpers import init_config


class SavedEmbeddings:
    def __init__(self, saved_embs: Dict):
        self.word_to_vec = saved_embs
        # pick arbitrary word
        self.emb_size = self.word_to_vec["the"].shape[0]
        self.zero_vector = numpy.zeros(self.emb_size, dtype=numpy.float32)
        self.mean_vec = numpy.mean(list(self.word_to_vec.values()))
        self.return_zero_for_oov = True

    def get(self, word):
        try:
            return self.word_to_vec[word]
        except KeyError:
            if self.return_zero_for_oov:
                return self.zero_vector
            # return numpy.random.uniform(-1., 1., self.emb_size)
            return self.mean_vec

    def __getitem__(self, word):
        return self.get(word)


def create_embeddings(
        pretrained_embeddings, vocab_result_file,
        embeddings_result_file):
    kv = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=True)
    with open(vocab_result_file) as f:
        vocab = set(f.read().split("\n"))

    word_to_vec = {}
    for word in vocab:
        if word in kv:
            word_to_vec[word] = kv[word]

    with open(embeddings_result_file, "wb") as f:
        pickle.dump(word_to_vec, f)


def read_embeddings(embeddings_result_file):
    with open(embeddings_result_file, "rb") as f:
        return SavedEmbeddings(pickle.load(f))


if __name__ == '__main__':
    create_embeddings(**init_config("data", create_embeddings))
