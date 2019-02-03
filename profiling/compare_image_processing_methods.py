import time

import mlflow
from sklearn.model_selection import ParameterGrid

from preprocess_images import prep_imgs


grid = {"mode": ["joblib", "threads", "processes", "data_loader"], "n_jobs": [1, 2, 4, 8, 10]}
grid = ParameterGrid(grid)

for params in grid:
    print(params)
    with mlflow.start_run() as r:
        mode, n_jobs = params["mode"], params["n_jobs"]
        start = time.time()
        mlflow.log_param("mode", mode)
        mlflow.log_param("n_jobs", n_jobs)
        prep_imgs("/data/train2014", mode, n_jobs)
        mlflow.log_metric("elapsed", time.time() - start)
