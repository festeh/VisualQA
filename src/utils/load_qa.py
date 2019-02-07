import json
import logging
from pathlib import Path
from typing import List, Union, Dict
import numpy as np
from fastprogress import progress_bar
from nltk import word_tokenize
from pandas import DataFrame

from src.utils.load_images import read_image


def preprocess_text(text: str):
    return word_tokenize(text.lower())


def read_data(data_path: Union[Path, str], data_type='train'):
    """Gets the VQ2 data"""
    # TODO: support test data
    data_path = Path(data_path)
    DATA_FILES = {'train': ["v2_mscoco_train2014_annotations.json", "v2_OpenEnded_mscoco_train2014_questions.json"],
                  'val': ["v2_mscoco_val2014_annotations.json", "v2_OpenEnded_mscoco_val2014_questions.json"]}
    answers_file, questions_file = DATA_FILES[data_type]
    questions = json.load((data_path / questions_file).open())['questions']
    answers = json.load((data_path / answers_file).open())["annotations"]
    logging.info("Readed raw data")
    return questions, answers


# def filter_data(data, condition=lambda x: True):
#     return [x for x in data if condition(x)]

# def merge_questions_answers(questions: List, answers: List, flatten=False):
#     """Combine questions and answers obtained from `read_data` into single list"""
#     id_to_question = {q['question_id']: q for q in questions}
#     qa_merged = [{**a, **id_to_question[a['question_id']]} for a in answers]
#     if flatten:
#         qa_merged_flat = []
#         for qa in qa_merged:
#             answers = qa.pop('answers')
#             for ans_info in answers:
#                 qa_merged_flat.append({
#                     "question": qa["question"],
#                     "preprocessed_question": qa.get("preprocessed_question", ""),
#                     "image_id": qa["image_id"],
#                     "answer": ans_info["answer"],
#                     "preprocessed_answer": ans_info.get("preprocessed_answer", ""),
#                     "confidence": ans_info["answer_confidence"]})
#         return qa_merged_flat
#     return qa_merged


def preprocess_questions_answers(
        questions: List, annotations: List, max_answers=None, only_one_word_answers=True) -> DataFrame:
    data = []
    id_to_question = {q['question_id']: q for q in questions}
    for ann in progress_bar(annotations):
        question_info = id_to_question[ann['question_id']]
        preprecessed_question = preprocess_text(question_info["question"])
        img_id = question_info["image_id"]
        for ans in ann["answers"]:
            if ans["answer_confidence"] != "no":
                data.append({"question": question_info["question"],
                             "preprocessed_question": preprecessed_question,
                             "question_id": question_info["question_id"],
                             "image_id": img_id,
                             "answer": ans["answer"]})
    data = DataFrame(data)

    # TODO: in future consider all (lost ~200k examples from 2mil)
    if only_one_word_answers:
        allowed_answers = data["answer"].str.split().str.len() == 1
        data = data[allowed_answers]

    if max_answers is not None:
        allowed_answers = data["answer"].value_counts()[:max_answers]
        data = data[data.answer.isin(allowed_answers)]

    return data


def process_part_qa_data(data_path: Union[Path, str], data_type='train', max_answers=None):
    logging.info(f"Reading {data_type} data part")
    q, a = read_data(data_path, data_type)
    return preprocess_questions_answers(q, a, max_answers=max_answers)


def save_qa_data(data, save_data_path, data_type="train"):
    save_dir = Path(save_data_path) / data_type
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    file_path = save_dir / "qa.pkl"
    data.to_pickle(file_path)


def filter_qa_pairs(qa: List[Dict], removed=("no",)):
    """Select only QA pairs in which annotators were more or less confident"""
    return [elem for elem in qa if elem["confidence"] not in removed]


def sample_examples(qa: List[Dict], img_path: Path, n_examples):
    """Firstly, randomly selects questions-answer pairs, then obtains corresponding images"""
    idxs = np.random.choice(len(qa), n_examples)
    if isinstance(qa, DataFrame):
        qs = [elem[1] for elem in qa.iloc[idxs].iterrows()]
        imgs = [read_image(img_path, id_['image_id'], True) for id_ in qs]
    else:
        qs = [qa[idx] for idx in idxs]
        imgs = [read_image(img_path, q['image_id'], True) for q in qs]
    return imgs, qs
