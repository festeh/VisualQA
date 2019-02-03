import logging
import pickle
import torch
from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.baseline import BaselineModel
from src.utils.load_all_data import get_data, VisualQADataset

config = {
    "raw_data_path": "/data/",
    "preprocessed_data_path": "/data/preprocessed"
}

LOG_FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')

data, ans_vocab = get_data(Path(config["raw_data_path"]), Path(config['preprocessed_data_path']),
                           parts=["train", "val"])
train_dataset = VisualQADataset(*data['train'], answer_vocabulary=ans_vocab)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
model = BaselineModel(
    question_emb_size=300,
    image_emb_size=4096,
    n_classes=1000,
    h_size=300,
    emb_path="/data/preprocessed/embeddings.pkl")
model.to(device)

optimizer = Adam(model.parameters(), 1e-3)
loss_fn = CrossEntropyLoss()

for questions, answers, image_embs in train_loader:
    optimizer.zero_grad()
    logits = model(questions, image_embs.to(device))
    loss = loss_fn(logits, answers.to(device))
    loss.backward()
    optimizer.step()
    print(loss.item())
