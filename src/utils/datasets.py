import json
import re
from functools import partial

import h5py
import numpy
import pandas as pd
from allennlp.data import Vocabulary, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from src.utils.helpers import init_config


class VisualQADataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _exctract_img_id(self, fname) -> int:
        return int(re.sub(r'^0+', '', fname.split("_")[-1]))

    def init_common(self, images_result_file,
                    qa_result_file,
                    filenames_result_file):
        self.qa = pd.read_pickle(qa_result_file)
        with open(filenames_result_file) as f:
            image_filenames = json.load(f)
        image_ids = [self._exctract_img_id(f) for f in image_filenames]
        self.image_id_to_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.index_to_image_filename = {idx: fname for idx, fname in enumerate(image_filenames)}

        self.preprocessed_imgs = h5py.File(images_result_file, "r")["images"]
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, text):
        text_tokens = [Token(w) for w in text.split()]
        return Instance({"question": TextField(text_tokens, self.token_indexers)})

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, item):
        raise NotImplementedError()


class VisualQATrainDataset(VisualQADataset):
    def __init__(
            self,
            train_images_result_file,
            train_qa_result_file,
            train_filenames_result_file,
            vocab_result_file):
        super().__init__()
        self.init_common(
            filenames_result_file=train_filenames_result_file,
            qa_result_file=train_qa_result_file,
            images_result_file=train_images_result_file)
        self.vocab = Vocabulary()
        self.vocab.set_from_file(filename=vocab_result_file, oov_token="[UNK]")
        possible_answers = self.qa.answer.value_counts()
        self.answer_vocabulary = {ans: idx for idx, ans in enumerate(possible_answers.index)}

    def __getitem__(self, idx):
        info = self.qa.iloc[idx]
        answer = info['answer']
        image = self.preprocessed_imgs[self.image_id_to_index[info['image_id']]]
        question = self.text_to_instance(info["preprocessed_question"])
        return question, image, self.answer_vocabulary[answer]


class VisualQAValDataset(VisualQADataset):

    def __init__(
            self,
            val_qa_result_file,
            val_images_result_file,
            val_filenames_result_file,
            vocab,
            answer_vocabulary):
        super().__init__()
        self.init_common(
            qa_result_file=val_qa_result_file,
            images_result_file=val_images_result_file,
            filenames_result_file=val_filenames_result_file)
        self.vocab = vocab
        self.answer_vocabulary = answer_vocabulary

    def __getitem__(self, idx):
        info = self.qa.iloc[idx]
        answers = info['answer']
        answer_idxs = [self.answer_vocabulary.get(ans, -1) for ans in answers]
        if len(answer_idxs) < 10:
            answer_idxs = answer_idxs + [-1] * (10 - len(answer_idxs))
        image = self.preprocessed_imgs[self.image_id_to_index[info['image_id']]]
        question = self.text_to_instance(info["preprocessed_question"])
        return question, image, numpy.array(answer_idxs)


def my_collate(batch, vocab):
    questions = Batch([x[0] for x in batch])
    questions.index_instances(vocab)
    rest = [x[1:] for x in batch]
    question_batch = questions.as_tensor_dict()["question"]["tokens"]
    image_batch, answer_batch = default_collate(rest)
    return [(question_batch, image_batch), answer_batch]


if __name__ == "__main__":
    data = VisualQATrainDataset(**init_config("data", VisualQATrainDataset.__init__))
    dl = DataLoader(data, batch_size=12, collate_fn=partial(my_collate, vocab=data.vocab))
    elem = next(iter(dl))
    print(elem)
