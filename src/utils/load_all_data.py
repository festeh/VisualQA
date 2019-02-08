import json
from logging import info
from pathlib import Path

import h5py
import os
import re

import numpy
from pandas import read_pickle
from torch.utils.data import Dataset

from src.utils.load_images import process_batch, load_pretrained_feature_extractor, read_image, preprocess_images
from src.utils.load_qa import process_part_qa_data, save_qa_data


class VisualQADataset(Dataset):
    def __init__(self, qa, preprocessed_images_file, image_filenames, answer_vocabulary):
        self.qa = qa
        image_ids = [self._exctract_img_id(f) for f in image_filenames]
        self.image_id_to_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.index_to_image_filename = {idx: fname for idx, fname in enumerate(image_filenames)}
        self.preprocessed_imgs = preprocessed_images_file
        self.answer_vocabulary = answer_vocabulary

    def _exctract_img_id(self, fname) -> int:
        return int(re.sub(r'^0+', '', fname.split("_")[-1]))

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, idx):
        info = self.qa.iloc[idx]
        question = info['preprocessed_question']
        answer = info['answer']
        answer_idx = self.answer_vocabulary[answer]
        image = self.preprocessed_imgs[self.image_id_to_index[info['image_id']]]
        return (question, image), answer_idx


class VisualQAValidationDataset(VisualQADataset):

    def __init__(self, qa, preprocessed_images_file, image_filenames, answer_vocabulary):
        super().__init__(qa, preprocessed_images_file, image_filenames, answer_vocabulary)

    def __getitem__(self, idx):
        info = self.qa.iloc[idx]
        question = info['preprocessed_question']
        answers = info['answer']
        answer_idxs = [self.answer_vocabulary.get(ans, -1) for ans in answers]

        if len(answer_idxs) < 10:
            answer_idxs = answer_idxs + [-1] * (10 - len(answer_idxs))

        image = self.preprocessed_imgs[self.image_id_to_index[info['image_id']]]
        return (question, image), numpy.array(answer_idxs)


class AnswerVocabulary:
    def __init__(self, unique_answers):
        self.answer_to_index = {ans: idx for idx, ans in enumerate(unique_answers)}
        self.index_to_answer = {v: k for k, v in self.answer_to_index.items()}

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.answer_to_index[item]
        elif isinstance(item, int):
            return self.index_to_answer[item]
        raise RuntimeError(f"Item should be str or int, got {type(item)}")

    def get(self, item, default_value):
        try:
            return self.__getitem__(item)
        except:
            return default_value


def get_data(raw_data_path: Path,
             preprocessed_data_path: Path, parts=("train", "val"), max_answers=1000):
    preprocessed_qa_path = preprocessed_data_path / "qa"
    preprocessed_image_dir = preprocessed_data_path / "images"

    qa_datasets = []
    image_datasets = []
    image_filenames = []

    for data_type in parts:
        qa_part_path = preprocessed_qa_path / data_type
        if not qa_part_path.exists():
            info(f"Preprocess qa data for {data_type} part")
            qa_data = process_part_qa_data(raw_data_path, data_type, max_answers=max_answers)
            save_qa_data(qa_data, preprocessed_qa_path, data_type)
        else:
            info(f"Read preprocessed qa data for {data_type} part")
            qa_data = read_pickle(preprocessed_qa_path / data_type / "qa.pkl")

        image_part_path = preprocessed_image_dir / data_type
        filenames_path = image_part_path / "paths.json"
        preprocessed_image_path = image_part_path / "prepared_imgs.hdf"
        if not image_part_path.exists():
            info(f"Preprocessing images for part {data_type}")
            filenames, images = preprocess_images(raw_data_path / f"{data_type}2014")
            image_part_path.mkdir(parents=True)
            json.dump(filenames, filenames_path.open("w"))
            with h5py.File(preprocessed_image_path, "w") as f:
                f.create_dataset("images", data=images, dtype='float32')
        else:
            info(f"Loading preprocessed images for part {data_type}")
            f = h5py.File(preprocessed_image_path, "r")
            images = f['images']
            filenames = json.load(filenames_path.open("r"))

        qa_datasets.append(qa_data)
        image_datasets.append(images)
        image_filenames.append(filenames)
        info("Done loading!")

    result = {data_type: [qa_datasets[idx], image_datasets[idx], image_filenames[idx]]
              for idx, data_type in enumerate(parts)}

    possible_answers = qa_datasets[0].answer.value_counts().index

    return result, AnswerVocabulary(possible_answers)


if __name__ == "__main__":
    qa_data, answers_vocab = get_data(Path(os.environ["RAW_DATA_PATH"]),
                                      Path(os.environ["PROCESSED_DATA_PATH"]), parts=["train", "val"])
    datasets, images, filenames = qa_data["train"]
    dataset = VisualQADataset(datasets, image_filenames=filenames, preprocessed_images_file=images)
    features = []
    images = []

    image_dir = Path(os.environ["IMAGE_DIR"])

    for idx in numpy.random.choice(len(dataset), 10):
        image_id = dataset.qa.iloc[idx]['image_id']
        features.append(dataset[idx][-1])
        img_filename = dataset.index_to_image_filename[dataset.image_id_to_index[image_id]]
        images.append(read_image(image_dir / f"{img_filename}.jpg", resize=True))
    ground_truth = process_batch(model=load_pretrained_feature_extractor(),
                                 imgs=numpy.stack(images),
                                 processed=[],
                                 already_tensor=False)
    print(numpy.stack(features) - ground_truth[0])
