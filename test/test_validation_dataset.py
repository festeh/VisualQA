from pathlib import Path

from junk.load_all_data import get_data
from src.utils.datasets import VisualQAValDataset

config = {
    "raw_data_path": "/data/",
    "preprocessed_data_path": "/data/preprocessed"
}
data, ans_vocab = get_data(Path(config["raw_data_path"]), Path(config['preprocessed_data_path']),
                           parts=["train", "val"])
dataset = VisualQAValDataset(*data["val"], answer_vocabulary=ans_vocab)
print(dataset[0])
