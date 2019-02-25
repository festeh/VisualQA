import logging
from functools import partial
import torch

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.metrics import VisualQAAccuracy
from src.models.model import BaselineModel
from src.utils.datasets import VisualQAValDataset, VisualQATrainDataset, my_collate
from src.utils.helpers import init_config, filter_config

LOG_FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')

experiment_config = init_config()
data_config = experiment_config.pop("data")
training_config = experiment_config.pop("training")
train_dataset = VisualQATrainDataset(**filter_config(data_config, VisualQATrainDataset.__init__))
vocab = train_dataset.vocab
# train_dataset.qa = train_dataset.qa[:1000]

val_dataset = VisualQAValDataset(
    **filter_config(data_config, VisualQAValDataset.__init__),
    vocab=vocab,
    answer_vocabulary=train_dataset.answer_vocabulary)

train_loader = DataLoader(
    train_dataset, batch_size=training_config.pop("train_batch_size"),
    shuffle=True, collate_fn=partial(my_collate, vocab=vocab))
val_loader = DataLoader(
    val_dataset, batch_size=training_config.pop("val_batch_size"),
    shuffle=False, collate_fn=partial(my_collate, vocab=vocab))

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

model = BaselineModel(
    config=experiment_config.pop("model"),
    embeddings_result_file=data_config.get("embeddings_result_file"),
    vocab=vocab)

optimizer = Adam(model.parameters(), training_config.pop("lr"))
loss = CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': VisualQAAccuracy(),
                                                 'nll': Loss(loss)
                                                 }, device=device)

run_avg = RunningAverage(output_transform=lambda x: x)
run_avg.attach(trainer, 'loss')

pbar = ProgressBar(persist=True, bar_format=None)
pbar.attach(trainer, ['loss'])

trainer.run(train_loader, max_epochs=training_config.pop("n_epochs"))
