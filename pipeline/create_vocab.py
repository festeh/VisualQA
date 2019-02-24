# Preprocess qa
from pathlib import Path

from pipeline.init import experiment_config
from src.utils.dvc import dvc_run
from src.utils.helpers import filter_config
from src.utils.vocab_creation import create_vocab

data_params = experiment_config.pop("data")
dvc_run("src/utils/vocab_creation.py",
        script_arguments=filter_config(data_params.as_dict(), create_vocab),
        dvc_file_name=f"{Path(__file__).stem}.dvc",
        input_deps=[data_params.get("train_qa_result_file"), data_params.get("val_qa_result_file")],
        output_deps=data_params.get("vocab_result_file"))
