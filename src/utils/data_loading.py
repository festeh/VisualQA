import json
from pathlib import Path
from typing import List, Union

from skimage.io import imread
from skimage.transform import resize as imresize


def read_data(data_path: Union[Path, str], data_type='train'):
    """Gets the VQ2 data"""
    # TODO: support test data
    data_path = Path(data_path)
    DATA_FILES = {'train': ["v2_mscoco_train2014_annotations.json", "v2_OpenEnded_mscoco_train2014_questions.json"],
                  'val': ["v2_mscoco_val2014_annotations.json", "v2_OpenEnded_mscoco_val2014_questions.json"]}
    answers_file, questions_file = DATA_FILES[data_type]
    questions = json.load((data_path / questions_file).open())['questions']
    answers = json.load((data_path / answers_file).open())["annotations"]
    return questions, answers


def merge_questions_answers(questions: List, answers: List, flatten=False):
    """Combine questions and answers obtained from `read_data` into single list"""
    id_to_question = {q['question_id']: q for q in questions}
    qa_merged = [{**a, **id_to_question[a['question_id']]} for a in answers]
    if flatten:
        qa_merged_flat = []
        for qa in qa_merged:
            answers = qa.pop('answers')
            for ans_info in answers:
                qa_merged_flat.append({**qa, **{"answer": ans_info["answer"],
                                                "confidence": ans_info["answer_confidence"]}})
        return qa_merged_flat
    return qa_merged


def read_image(path, image_id: int, resize=False):
    path = Path(path)
    image_id = str(image_id)
    padded_image_id = "0" * (12 - len(image_id)) + image_id
    image_path = path / f"COCO_train2014_{padded_image_id}.jpg"
    image = imread(image_path)
    if resize:
        return imresize(image, (224, 224), mode='reflect', anti_aliasing=True)
    return image
