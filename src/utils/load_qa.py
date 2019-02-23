import json
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import numpy as np
from nltk import word_tokenize
from pandas import DataFrame
from tqdm import tqdm
import click


def preprocess_text(text: str) -> str:
    words = word_tokenize(text.lower())
    return ' '.join(words)


DATA_FILES = {'train': ["v2_mscoco_train2014_annotations.json", "v2_OpenEnded_mscoco_train2014_questions.json"],
              'val': ["v2_mscoco_val2014_annotations.json", "v2_OpenEnded_mscoco_val2014_questions.json"],
              'sample': ["annotations.json", "questions.json"]}


def read_questions_answers(data_path: Union[Path, str], data_type='train') -> Tuple[List, List]:
    """Gets the VQ2 data"""
    # TODO: support test data
    data_path = Path(data_path)

    answers_file, questions_file = DATA_FILES[data_type]
    questions = json.load((data_path / questions_file).open())['questions']
    answers = json.load((data_path / answers_file).open())["annotations"]
    logging.info("Loaded raw questions and answers")
    return questions, answers


def preprocess_questions_answers(
        questions: List,
        annotations: List,
        max_answers=None,
        only_one_word_answers=True,
        flatten=False) -> DataFrame:
    data = []
    id_to_question = {q['question_id']: q for q in questions}
    for ann in tqdm(annotations):
        question_info = id_to_question[ann['question_id']]
        preprocessed_question = preprocess_text(question_info["question"])
        img_id = question_info["image_id"]

        confident_answers = [ans for ans in ann["answers"] if ans["answer_confidence"] != "no"]
        datum = {"question": question_info["question"],
                 "preprocessed_question": preprocessed_question,
                 "question_id": question_info["question_id"],
                 "image_id": img_id}
        if not flatten:
            for ans in confident_answers:
                data.append({**datum, "answer": ans["answer"]})
        else:
            data.append({**datum, **{"answer": [ans["answer"] for ans in confident_answers]}})
    data = DataFrame(data)

    # TODO: in future consider all (lost ~200k examples from 2mil)
    if only_one_word_answers:
        allowed_answers_idx = data["answer"].str.split().str.len() == 1
        data = data[allowed_answers_idx]

    if max_answers is not None:
        allowed_answers = data["answer"].value_counts()[:max_answers]
        data = data[data.answer.isin(allowed_answers.index)]

    return data


def process_part_qa_data(data_path: Union[Path, str], data_type='train', max_answers=None):
    logging.info(f"Reading {data_type} data part")
    q, a = read_questions_answers(data_path, data_type)
    if data_type == "train":
        return preprocess_questions_answers(q, a, max_answers=max_answers, flatten=False)
    else:
        return preprocess_questions_answers(
            q, a, max_answers=None, flatten=True, only_one_word_answers=False)


def save_qa_data(data, save_data_path, data_type="train"):
    save_dir = Path(save_data_path)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    file_path = save_dir / f"qa_{data_type}.pkl"
    data.to_pickle(file_path)


def filter_qa_pairs(qa: List[Dict], removed=("no",)):
    """Select only QA pairs in which annotators were more or less confident"""
    return [elem for elem in qa if elem["confidence"] not in removed]


def sample_examples(qa: List[Dict], img_path: Path, n_examples):
    """Firstly, randomly selects questions-answer pairs, then obtains corresponding images"""
    from src.utils.load_images import read_image
    idxs = np.random.choice(len(qa), n_examples)
    if isinstance(qa, DataFrame):
        qs = [elem[1] for elem in qa.iloc[idxs].iterrows()]
        imgs = [read_image(img_path, id_['image_id'], True) for id_ in qs]
    else:
        qs = [qa[idx] for idx in idxs]
        imgs = [read_image(img_path, q['image_id'], True) for q in qs]
    return imgs, qs


@click.command()
@click.option("--data_dir", help="path to qa data")
@click.option("--saving_dir", help="path to saved data")
@click.option("--max_answers", type=int, help="how many answers to use")
def main(data_dir, saving_dir, max_answers=1000):
    saving_dir = Path(saving_dir)
    if not saving_dir.exists():
        saving_dir.mkdir(parents=True)
    train_data = preprocess_questions_answers(*read_questions_answers(data_dir, "train"),
                                              max_answers=max_answers,
                                              only_one_word_answers=True)
    val_data = preprocess_questions_answers(*read_questions_answers(data_dir, "val"),
                                            max_answers=None, only_one_word_answers=False, flatten=True)
    save_qa_data(train_data, saving_dir, "train")
    save_qa_data(val_data, saving_dir, "val")


if __name__ == "__main__":
    main()
