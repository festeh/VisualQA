import json
from pathlib import Path

import numpy as np

from src.utils.load_qa import read_questions_answers, DATA_FILES


def sample_qa(questions, annotations, n_samples=100):
    np.random.seed(0)
    sample_annotation_idxs = np.random.choice(len(annotations), n_samples)
    sample_annotations = np.array(annotations)[sample_annotation_idxs]
    sample_questions = []
    for ann in sample_annotations:
        sample_questions.append([q for q in questions
                                 if q["question_id"] == ann["question_id"]][0])
    return sample_questions, list(sample_annotations)


if __name__ == '__main__':
    sample_q, sample_a = sample_qa(*read_questions_answers("data", "train"))
    ann_path, q_path = DATA_FILES["sample"]
    print(ann_path, q_path)
    root_dir = Path("data")
    json.dump({"questions": sample_q}, (root_dir / q_path).open("w"))
    json.dump({"annotations": sample_a}, (root_dir / ann_path).open("w"))
