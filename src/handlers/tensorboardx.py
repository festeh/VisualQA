from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self):
        self.writer = SummaryWriter()

    def __call__(self, engine, mode, evaluator=None):
        if mode == "train":
            self.writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        else:
            accuracy = evaluator.state.metrics['accuracy']
            self.writer.add_scalar("eval/acc", accuracy, engine.state.epoch)
