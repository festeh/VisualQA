import mlflow
from ignite.engine import Events


class MlflowHandler:
    def __init__(self):
        self.run = mlflow.start_run()
        mlflow.log_artifact("config.jsonnet")

    def __call__(self, engine, event, evaluator=None):
        if event == Events.EPOCH_COMPLETED:
            accuracy = evaluator.state.metrics['accuracy']
            mlflow.log_metric("accuracy", accuracy)
        elif event == Events.COMPLETED:
            mlflow.end_run(self.run)
