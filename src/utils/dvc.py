import logging
import subprocess, os
from typing import Dict, Union, List

# TODO: nasty, use env variable?
CONDA_PATH = "/home/dima/miniconda3/envs/visualqa/"


def dvc_run(
        script_path: str,
        script_arguments: Dict,
        output_deps: Union[str, List[str]],
        dvc_file_name=None):
    arguments = " ".join(f"--{arg_name}={arg_value}" for arg_name, arg_value in script_arguments.items())
    if isinstance(output_deps, str):
        output_deps = [output_deps]
    output_deps = ' '.join([f'-o {dep}' for dep in output_deps])
    dvc_file_name = f'-f {dvc_file_name}' if dvc_file_name is not None else ''
    command = f"dvc run --overwrite-dvcfile {dvc_file_name} {output_deps} -d {script_path} python {script_path} {arguments}"
    logging.info(f"Executing: {command}")
    execute_dvc_command(command)


def execute_dvc_command(dvc_command):
    subprocess.run(f'bash -c "conda run -p {CONDA_PATH} {dvc_command}"', shell=True)
