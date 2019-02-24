from pathlib import Path
from typing import Dict, Callable

from allennlp.common import Params


def create_parent_dir_if_not_exists(file_path: Path):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)


def filter_config(config: Dict, function: Callable):
    func_args = set(function.__code__.co_varnames)
    return {k: v for k, v in config.items() if k in func_args}


def init_config(config_part: str, target_function: Callable):
    data_params = Params.from_file("config.jsonnet").pop(config_part).as_dict()
    return filter_config(data_params, target_function)
