# Preprocess qa
from pathlib import Path

from pipeline.init import experiment_config
from src.utils.dvc import dvc_run
from src.utils.helpers import filter_config
from src.data_preprocessing.pretrained_embeddings import create_embeddings

data_params = experiment_config.pop("data")
dvc_run("src/data_preprocessing/pretrained_embeddings.py",
        script_arguments=filter_config(data_params.as_dict(), create_embeddings),
        dvc_file_name=f"{Path(__file__).stem}.dvc",
        input_deps=[data_params.get("vocab_result_file")],
        output_deps=data_params.get("embeddings_result_file"))
