import os
import shutil
import tempfile

import mlflow
from ignite.engine import Events, Engine


class MlflowHandler:
    def __init__(self, evaluator):
        self.run = mlflow.start_run()
        self.evaluator = evaluator

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.STARTED, self.save_config)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_metric)
        engine.add_event_handler(Events.COMPLETED, self.end_run)

    def log_metric(self, engine):
        accuracy = self.evaluator.state.metrics['accuracy']
        mlflow.log_metric("accuracy", accuracy)

    def end_run(self, engine):
        mlflow.end_run()

    def save_config(self, engine):
        # Convinience for artifact ui...
        shutil.copy("config.jsonnet", "/tmp/config.json")
        mlflow.log_artifact("/tmp/config.json")
