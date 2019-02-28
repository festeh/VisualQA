import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict, Tuple
import numpy as np
from nltk import word_tokenize
from pandas import DataFrame
from tqdm import tqdm

from src.utils.helpers import init_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    words = word_tokenize(text.lower())
    return ' '.join(words)


def read_questions_answers(questions_path, answers_path) -> Tuple[List, List]:
    """Gets the VQ2 questions and answers"""
    with open(questions_path) as f:
        questions = json.load(f)['questions']
    with open(answers_path) as f:
        answers = json.load(f)["annotations"]
    logger.info("Loaded raw questions and answers")
    return questions, answers


def preprocess_part_questions_answers(
        questions: List,
        annotations: List) -> List:
    data = []
    id_to_question = {q['question_id']: q for q in questions}
    for ann in tqdm(annotations):
        question_info = id_to_question[ann['question_id']]
        question_id = question_info["question_id"]
        img_id = question_info["image_id"]
        question_text = question_info["question"]
        preprocessed_question = preprocess_text(question_text)

        datum = {"question": question_text,
                 "preprocessed_question": preprocessed_question,
                 "question_id": question_id,
                 "image_id": img_id}

        all_answers = ann["answers"]
        confident_answers = [preprocess_text(ans["answer"]) for ans in all_answers
                             if ans["answer_confidence"] != "no"]

        datum["answers"] = confident_answers
        data.append(datum)
    return data


def filter_qa_pairs(qa_data, max_question_length=None, max_answers=None, answer_vocab_result_file=None):
    """Select only QA pairs in which annotators were more or less confident"""
    logging.info(f"Filtering data")
    if max_answers is not None:
        logging.info(f"Using only {max_answers} possible answers")
        answer_counts = Counter()
        # build and save answer vocab
        for qa in qa_data:
            answer_counts.update(qa["answers"])
        allowed_answers = [ans for ans, _ in answer_counts.most_common(max_answers)]
        with open(answer_vocab_result_file, "w") as f:
            answer_vocab = {ans: idx for idx, ans in enumerate(allowed_answers)}
            json.dump(answer_vocab, f)
        # filter answers
        allowed_answers = set(allowed_answers)
        for qa in qa_data:
            qa["answers"] = [ans for ans in qa["answers"] if ans in allowed_answers]

    return qa_data


def save_qa_data(data, save_data_path):
    save_data_path = Path(save_data_path)
    if not save_data_path.parent.exists():
        save_data_path.parent.mkdir(parents=True)
    with save_data_path.open("wb") as f:
        pickle.dump(data, f)


# def sample_examples(qa: List[Dict], img_path: Path, n_examples):
#     """Firstly, randomly selects questions-answer pairs, then obtains corresponding images"""
#     from src.data_preprocessing.load_images import read_image
#     idxs = np.random.choice(len(qa), n_examples)
#     if isinstance(qa, DataFrame):
#         qs = [elem[1] for elem in qa.iloc[idxs].iterrows()]
#         imgs = [read_image(img_path, id_['image_id'], True) for id_ in qs]
#     else:
#         qs = [qa[idx] for idx in idxs]
#         imgs = [read_image(img_path, q['image_id'], True) for q in qs]
#     return imgs, qs


def preprocess_questions_answers(
        train_annotations, val_annotations,
        train_questions, val_questions,
        train_qa_result_file, val_qa_result_file,
        answer_vocab_result_file,
        max_answers):
    train_data = preprocess_part_questions_answers(*read_questions_answers(train_questions, train_annotations))
    val_data = preprocess_part_questions_answers(*read_questions_answers(val_questions, val_annotations))
    train_data = filter_qa_pairs(
        train_data,
        max_answers=max_answers,
        answer_vocab_result_file=answer_vocab_result_file)
    save_qa_data(train_data, train_qa_result_file)
    save_qa_data(val_data, val_qa_result_file)


if __name__ == "__main__":
    preprocess_questions_answers(**init_config("data", preprocess_questions_answers))
