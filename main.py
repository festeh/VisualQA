import logging
from functools import partial
from os import environ

import torch
from ignite.contrib.handlers import ProgressBar, LRScheduler
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import RunningAverage
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.handlers.eval import EvalHandler
from src.handlers.mlflow import MlflowHandler
from src.handlers.tensorboardx import TensorboardHandler
from src.metrics import VisualQAAccuracy
from src.models.model import BaselineModel
from src.utils.datasets import VisualQAValDataset, VisualQATrainDataset, my_collate
from src.utils.helpers import init_config, filter_config, get_experiment_name

LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')

DEBUGGING_MODE = int(environ.get("DEBUG_MODE", 0)) == 1
if DEBUGGING_MODE:
    logging.info("Run was started in debugging mode: no info will be stored in mlflow or tensorboard")
else:
    logging.info("Run was started in normal mode: info will be stored in mlflow and tensorboard")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

experiment_name = get_experiment_name()
experiment_config = init_config()
data_config = experiment_config.pop("data")
training_config = experiment_config.pop("training")
train_dataset = VisualQATrainDataset(
    **filter_config(data_config, VisualQATrainDataset.__init__))
vocab = train_dataset.vocab

val_dataset = VisualQAValDataset(
    **filter_config(data_config, VisualQAValDataset.__init__),
    vocab=vocab,
    answer_vocabulary=train_dataset.answer_vocabulary)

train_loader = DataLoader(
    train_dataset, batch_size=training_config.pop("train_batch_size"),
    shuffle=True, collate_fn=partial(my_collate, vocab=vocab), num_workers=6)
val_loader = DataLoader(
    val_dataset, batch_size=training_config.pop("val_batch_size"),
    shuffle=False, collate_fn=partial(my_collate, vocab=vocab), num_workers=6)

model = BaselineModel(
    config=experiment_config.pop("model"),
    embeddings_result_file=data_config.get("embeddings_result_file"),
    vocab=vocab)

optimizer = Adam(model.parameters(), training_config.pop("lr"))
loss = BCEWithLogitsLoss()

trainer = create_supervised_trainer(model, optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model, metrics={'accuracy': VisualQAAccuracy()}, device=device)

# create and add handlers
run_avg = RunningAverage(output_transform=lambda x: x)
run_avg.attach(trainer, 'loss')
pbar = ProgressBar(persist=False, bar_format=None)
pbar.attach(trainer, ['loss'])
pbar.attach(evaluator)
eval_handler = EvalHandler(evaluator=evaluator, data_loader=val_loader)
eval_handler.attach(trainer)
if not DEBUGGING_MODE:
    run_name = environ.get("RUN_NAME", "NOTSET")
    tb_handler = TensorboardHandler(experiment_name=experiment_name, evaluator=evaluator)
    tb_handler.attach(trainer)
    mlflow_handler = MlflowHandler(run_name=run_name, experiment_name=experiment_name, evaluator=evaluator)
    mlflow_handler.attach(trainer)


# finally run training process
trainer.run(train_loader, max_epochs=training_config.pop("n_epochs"))
