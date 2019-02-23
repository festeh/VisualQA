from src.utils.load_qa import main
from pipeline.init import experiment_config

data_config = experiment_config.pop("data").as_dict()
main.callback(**data_config)