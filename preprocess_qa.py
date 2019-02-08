import mlflow
import click
from pathlib import Path

import json

from src.utils.load_qa import preprocess_questions_answers


# @click.command(help="Preprocesses VisualQA v.2 data")
# @click.option("--data_path")
# def preprocess_questions_answers(data_path, data_type="train"):
#     data_path = Path(data_path)
#     train_questions, train_answers = read_data(data_path, data_type)
#     train_data = create_train_data(train_questions, train_answers)
    # save_dir = data_path / "preprocessed"
    # if not save_dir.exists():
    #     save_dir.mkdir()
    # train_path = save_dir / "qa_train.pkl"
    # train_data.to_pickle(train_path)
    # mlflow.log_artifact(train_path, "qa_train")


if __name__ == '__main__':
    import plac; plac.call(preprocess_questions_answers)
