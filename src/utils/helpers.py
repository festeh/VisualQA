from pathlib import Path
from typing import Dict, Callable

from allennlp.common import Params


def create_parent_dir_if_not_exists(file_path: Path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)


def filter_config(config: Dict, function: Callable):
    func_args = set(function.__code__.co_varnames)
    return {k: v for k, v in config.items() if k in func_args}


def init_config(config_part: str = None, target_function: Callable = None):
    data_params = Params.from_file("config.jsonnet")
    if config_part is not None:
        data_params = data_params.pop(config_part)
    if target_function is None:
        return data_params
    return filter_config(data_params.as_dict(), target_function)


def get_experiment_name() -> str:
    from pygit2 import Repository
    experiment_name = Repository('.').head.shorthand
    if experiment_name == "master":
        return "Default"
    return experiment_name
