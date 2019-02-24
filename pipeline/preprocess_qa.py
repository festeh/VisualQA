# Preprocess qa
from pathlib import Path

from pipeline.init import experiment_config
from src.utils.dvc import dvc_run

data_params = experiment_config.pop("data")
dvc_run("src/utils/load_qa.py",
        dvc_file_name=f"{Path(__file__).stem}.dvc",
        output_deps=[data_params.get("train_qa_result_file"), data_params.get("val_qa_result_file")])
