# experiments.py

import mlflow
import mlflow.sklearn
import json


class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name=None):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        return self.run

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, sensor_id):
        # Log metrics with sensor-specific keys
        for key, value in metrics.items():
            mlflow.log_metric(f"{sensor_id}_{key}", value)

    def log_model(self, model, sensor_id, model_name="model"):
        mlflow.sklearn.log_model(model, f"{sensor_id}_{model_name}")

    def log_artifact(self, artifact_path):
        mlflow.log_artifact(artifact_path)

    def end_run(self):
        mlflow.end_run()
