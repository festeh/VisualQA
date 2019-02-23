
import subprocess, os

# TODO: nasty, use env variable?
from typing import Dict, Union, List

CONDA_PATH = "/home/dima/miniconda3/envs/visualqa/"


def dvc_run(
        script_path: str,
        script_arguments: Dict,
        output_deps=Union[str, List[str]]):
    arguments_str = " ".join(f"--{arg_name}={arg_value}" for arg_name, arg_value in script_arguments)
    if isinstance(output_deps, str):
        output_deps = [output_deps]

    output_deps_str = [f"-o {dep}" for dep in output_deps]
    command = f"dvc run {output_deps_str} -d {script_path} python {script_path} {arguments_str}"
    return command


def execute_dvc_command(dvc_command):
    subprocess.run(f'bash -c "conda run -p {CONDA_PATH} {dvc_command}"', shell=True)



