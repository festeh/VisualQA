import logging
import pickle

import numpy
import torch
from pathlib import Path

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, Accuracy
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.baseline import BaselineModel
from src.utils.load_all_data import get_data, VisualQADataset, VisualQAValidationDataset

config = {
    "raw_data_path": "/data/",
    "preprocessed_data_path": "/data/preprocessed"
}

LOG_FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')

data, ans_vocab = get_data(Path(config["raw_data_path"]), Path(config['preprocessed_data_path']),
                           parts=["train", "val"])
train_dataset = VisualQADataset(*data['train'], answer_vocabulary=ans_vocab)

#TODO: remove
train_dataset.qa = train_dataset.qa[:1000]

val_dataset = VisualQAValidationDataset(*data['val'], answer_vocabulary=ans_vocab)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
model = BaselineModel(
    question_emb_size=300,
    image_emb_size=4096,
    n_classes=1000,
    h_size=300,
    emb_path="/data/preprocessed/embeddings.pkl").to(device)

optimizer = Adam(model.parameters(), 1e-3)
loss = CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': Accuracy(),
                                                 'nll': Loss(loss)
                                                 }, device=device)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")


# trainer.run(train_loader, max_epochs=100)


# #
# for iter, ((questions, image_embs), answers) in tqdm(enumerate(train_loader)):
#     optimizer.zero_grad()
#     logits = model(questions, image_embs.to(device))
#     loss = loss_fn(logits, answers.to(device))
#     loss.backward()
#     optimizer.step()
#     print(loss.item())
#
#     if iter % 100 == 0:
#         accs = []
#         for val_iter, (val_questions, val_answers, val_image_embs) in tqdm(enumerate(val_loader, 1)):
#             with torch.no_grad():
#                 logits = model(val_questions, val_image_embs.to(device))
#                 preds = logits.argmax(1).cpu()
#             n_correct_for_question = (val_answers == preds[:, None]).sum(1)
#             acc = n_correct_for_question.type(torch.float32) / 3
#             acc.clamp_(0, 1)
#             accs.extend(acc.numpy())
#
#             if val_iter % 100 == 0:
#                 print("Mean validation acuracy:", numpy.mean(accs))
#                 break
#
# torch.save(model.state_dict(), "models/baseline.pth")
