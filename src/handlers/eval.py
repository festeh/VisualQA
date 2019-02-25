from ignite.engine import Engine, Events


class EvalHandler:
    def __init__(self, evaluator, data_loader):
        self.evauator = evaluator
        self.data_loader = data_loader

    def run_evaluator(self, engine):
        self.evauator.run(self.data_loader)

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.run_evaluator)
