import numpy
import pandas
import pickle
from pathlib import Path

from gensim.models import KeyedVectors


def preprocess_w2v(path_to_w2v, save_path):
    words = set()
    for part in ["train", "val"]:
        df = pandas.read_pickle(f"/data/preprocessed/qa/{part}/qa.pkl")
        res = ' '.join(df.preprocessed_question.str.join(' ').values)
        words = words | set(res.split())
    vectors = KeyedVectors.load_word2vec_format(path_to_w2v, binary=True)
    word2vec = {}
    for w in words:
        try:
            word2vec[w] = vectors[w]
        except:
            pass
    pickle.dump(word2vec, Path(save_path).open("wb"))


def get_mean_w2v_vector(question: str, vectors: dict, emb_size: int):
    words = question.split()
    zero_vec = numpy.zeros(emb_size)
    valid_words = [w for w in words if w in vectors]
    if not valid_words:
        return zero_vec
    return numpy.mean([vectors[w] for w in valid_words], axis=0)



if __name__ == "__main__":
    vectors = pickle.load(Path("/data/preprocessed/embeddings.pkl").open("rb"))
    res = get_mean_w2v_vector("hi", vectors, 300)
    assert numpy.array_equal(res, vectors["hi"])
