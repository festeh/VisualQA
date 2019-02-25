from ignite.engine import Engine, Events
from tensorboardX import SummaryWriter


class TensorboardHandler:
    def __init__(self, evaluator):
        self.writer = SummaryWriter()
        self.evaluator = evaluator

    def add_loss(self, engine):
        self.writer.add_scalar("training_loss", engine.state.output, engine.state.iteration)

    def add_accuracy(self, engine):
        accuracy = self.evaluator.state.metrics['accuracy']
        self.writer.add_scalar("eval/acc", accuracy, engine.state.epoch)

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.add_loss)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.add_accuracy)
