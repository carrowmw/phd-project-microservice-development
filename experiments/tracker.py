# experiments/tracker.py

import mlflow
import mlflow.pytorch
from utils.config_helper import (
    get_batch_size,
    get_learning_rate,
    get_epochs,
    get_criterion,
    get_optimiser,
    get_model_type,
)


class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_id = mlflow.create_experiment(self.experiment_name)
        self.model_type = get_model_type()
        self.batch_size = get_batch_size()
        self.learning_rate = get_learning_rate()
        self.epochs = get_epochs()
        self.criterion = get_criterion()
        self.optimiser = get_optimiser()

    def start_parent_run(self):
        with mlflow.start_run(
            run_name="PARENT_RUN",
            experiment_id=self.experiment_id,
            tags={"version": "v1", "priority": "P1"},
            description="parent",
        ) as parent_run:
            return parent_run

    def start_child_run(self, run_name):
        with mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            nested=True,
        ) as child_run:
            return child_run

    def log_params(self):

        params = {
            "model_type": self.model_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "criterion": self.criterion,
            "optimiser": self.optimiser,
        }

        mlflow.log_params(params)

        # print(f"Logging model {loader[0]} to MLflow...")
        # mlflow.pytorch.log_model(model, "models")

        # # Register the model
        # model_name = f"sensor_model_{loader[0]}"
        # model_uri = f"runs:/{child_run.info.run_id}/models/{loader[0]}"
        # mlflow.register_model(model_uri, model_name)

    def log_model(self, model, model_name="model"):
        mlflow.pytorch.log_model(model, model_name)

    def register_model(self, sensor_name):
        model_name = f"sensor_model_{sensor_name}"
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
