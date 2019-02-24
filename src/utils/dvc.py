import logging
import subprocess, os
from typing import Dict, Union, List

# TODO: nasty, use env variable?
CONDA_PATH = "/home/dima/miniconda3/envs/visualqa/"


def dvc_run(
        script_path: str,
        output_deps: Union[str, List[str]],
        input_deps: Union[str, List[str]] = "",
        script_arguments=None,
        dvc_file_name=None):
    if script_arguments is not None:
        script_arguments = " ".join(f"--{arg_name}={arg_value}" for arg_name, arg_value in script_arguments.items())
    else:
        script_arguments = ""
    if isinstance(output_deps, str):
        output_deps = [output_deps]
    output_deps = ' '.join([f'-o {dep}' for dep in output_deps])
    if input_deps is None:
        input_deps = ""
    elif isinstance(input_deps, str):
        input_deps = [input_deps]
    input_deps = ' '.join([f'-d {dep}' for dep in input_deps])
    dvc_file_name = f'-f {dvc_file_name}' if dvc_file_name is not None else ''
    command = f"dvc run --overwrite-dvcfile {dvc_file_name} {output_deps} -d {script_path} {input_deps} " \
        f"python {script_path} {script_arguments}"
    logging.info(f"Executing: {command}")
    execute_dvc_command(command)


def execute_dvc_command(dvc_command):
    subprocess.run(f'bash -c "conda run -p {CONDA_PATH} {dvc_command}"', shell=True)
