# Preprocess qa
from pathlib import Path

from pipeline.init import experiment_config
from src.utils.dvc import dvc_run
from src.utils.helpers import filter_config
from src.utils.load_images import preprocess_images

data_params = experiment_config.pop("data")
dvc_run("src/utils/load_images.py",
        script_arguments=filter_config(data_params.as_dict(), preprocess_images),
        dvc_file_name=f"{Path(__file__).stem}.dvc",
        output_deps=[data_params.get("train_images_result_file"),
                     data_params.get("val_images_result_file"),
                     data_params.get("train_filenames_result_file"),
                     data_params.get("val_filenames_result_file")])
