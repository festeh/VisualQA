from itertools import chain
from pathlib import Path

import pandas as pd
from allennlp.common import Params

from src.utils.helpers import filter_config, create_parent_dir_if_not_exists


def create_vocab(
        train_qa_result_file, val_qa_result_file,
        vocab_result_file):
    qa_train = pd.read_pickle(train_qa_result_file)
    qa_val = pd.read_pickle(val_qa_result_file)
    combined = pd.concat([qa_train, qa_val])
    question_tokens = set(" ".join(combined.preprocessed_question.values).split())
    train_ans_tokens = set(qa_train.answer.values)
    val_ans_tokens = set(" ".join(chain(*qa_val.answer.values)).split())

    vocab = ["[PAD]", "[UNK]", "[EOS]"] + sorted(question_tokens | train_ans_tokens | val_ans_tokens)
    vocab_result_file = Path(vocab_result_file)
    create_parent_dir_if_not_exists(vocab_result_file)
    with vocab_result_file.open("w") as f:
        print("\n".join(vocab), file=f)
    return


if __name__ == "__main__":
    data_params = Params.from_file("config.jsonnet").pop("data").as_dict()
    data_params = filter_config(data_params, create_vocab)
    create_vocab(**data_params)
