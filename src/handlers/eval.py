class EvalHandler:
    def __init__(self, evaluator, data_loader):
        self.evauator = evaluator
        self.data_loader = data_loader

    def __call__(self, engine):
        self.evauator.run(self.data_loader)
