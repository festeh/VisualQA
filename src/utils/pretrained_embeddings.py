import pickle

import numpy
from gensim.models import KeyedVectors

from src.utils.helpers import init_config


class SavedEmbeddings:
    def __init__(self, vocab, keyed_vectors):
        self.vocab = vocab
        self.emb_size = keyed_vectors.vector_size
        self.zero_vector = numpy.zeros(self.emb_size, dtype=numpy.float32)
        self.word_to_vec = {}
        self._init_word_to_vec(keyed_vectors)
        self.return_zero_for_oov = True

    def _init_word_to_vec(self, keyed_vectors):
        for word in self.vocab:
            if word in keyed_vectors:
                self.word_to_vec[word] = keyed_vectors[word]

    def get(self, word):
        try:
            return self.word_to_vec[word]
        except KeyError:
            if self.return_zero_for_oov:
                return self.zero_vector
            return numpy.random.uniform(-1., 1., self.emb_size)

    def __getitem__(self, word):
        return self.get(word)


def create_embeddings(
        pretrained_embeddings, vocab_result_file,
        embeddings_result_file):
    kv = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=True)
    with open(vocab_result_file) as f:
        vocab = set(f.read().split("\n"))
    embeddings = SavedEmbeddings(vocab, kv)
    with open(embeddings_result_file, "wb") as f:
        pickle.dump(embeddings, f)


def read_embeddings(embeddings_result_file):
    with open(embeddings_result_file, "rb") as f:
        return pickle.load(f)


if __name__ == '__main__':
    create_embeddings(**init_config("data", create_embeddings))
