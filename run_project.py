import mlflow
from mlflow.utils import cli_args
import click


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@cli_args.NO_CONDA
def run_my_project(no_conda):
    mlflow.projects.run(".", use_conda=not no_conda, entry_point="preprocess_imgs",
                        parameters={"mode": "joblib", "n_jobs": 1})


if __name__ == '__main__':
    run_my_project()
