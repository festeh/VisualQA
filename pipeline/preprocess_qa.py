# Preprocess qa
from pathlib import Path

from pipeline.init import experiment_config
from src.utils.dvc import dvc_run

data_params = experiment_config.pop("data")
dvc_run("src/utils/load_qa.py",
        script_arguments=data_params.as_dict(),
        dvc_file_name=f"{Path(__file__).stem}.dvc",
        output_deps=data_params.get("qa_saving_dir"))
