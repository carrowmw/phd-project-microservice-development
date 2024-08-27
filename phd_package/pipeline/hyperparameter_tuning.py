# phd_package/pipeline/hyperparameter_tuning.py
import os
import mlflow
from typing import List, Tuple, Dict
import json
import logging
from datetime import datetime
import numpy as np
import optuna
from phd_package.config.paths import PIPELINE_CONFIG_PATH, TUNING_OUTPUT_DIR
from phd_package.utils.pipeline_helper import generate_random_string
from .pipeline_generator import Pipeline


class HyperparameterTuner:
    """
    Hyperparameter tuner class that uses Optuna to optimize the hyperparameters of the pipeline.
    """

    def __init__(
        self,
        config_path: str,
        optimize_metric: str = "r2",
        output_dir: str = TUNING_OUTPUT_DIR,
    ):
        self.config_path = config_path
        self.optimize_metric = optimize_metric
        self.output_dir = output_dir
        self.setup_logging()

    def setup_logging(self):
        # Create ouptut directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a unique filename for this turning session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"tuning_{timestamp}.log")
        self.results_file = os.path.join(self.output_dir, f"tuning_{timestamp}.json")

        # Setup the logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create file and console handlers (both will log INFO level messages)
        file_handler = logging.FileHandler(self.log_file)
        console_handler = logging.StreamHandler()

        # Create a formatter and set it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Capture Optuna's output
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.addHandler(file_handler)
        optuna_logger.addHandler(console_handler)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function to minimize the loss or maximize the accuracy score.
        """
        # Define the hyperparameters to tune
        experiment_name = f"Trail_{trial.number}_{generate_random_string(8)}"
        experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id):

            config = {
                "window_size": trial.suggest_categorical(
                    "window_size", [4, 8, 12, 16, 20, 24]
                ),
                "horizon": trial.suggest_categorical("horizon", [24]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                "epochs": trial.suggest_categorical("epochs", [5, 10, 20]),
                "model_type": trial.suggest_categorical("model_type", ["lstm"]),
                "hidden_dim": trial.suggest_categorical(
                    "hidden_dim", [32, 64, 128, 256]
                ),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                # LR Scheduler parameters
                "scheduler_step_size": trial.suggest_int("scheduler_step_size", 1, 10),
                "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.1, 0.9),
            }

            # Log the hyperparameters to MLflow
            mlflow.log_params(config)

            try:
                # Update the pipeline configuration with the trial's hyperparameters
                self.update_config_file(config)

                # Run the pipeline with the current hyperparameters
                pipeline = Pipeline(experiment_name=experiment_name)
                test_metrics_list = pipeline.run_pipeline()

                # Calculate the average performance across all models
                test_loss, test_mape, test_rmse, test_r2 = (
                    self.extract_average_performance(test_metrics_list)
                )

                # Log the results
                logging.info(
                    "Trial %d: loss=%.4f, mape=%.4f, rmse=%.4f, r2=%.4f",
                    trial.number,
                    np.mean(test_loss),
                    np.mean(test_mape),
                    np.mean(test_rmse),
                    np.mean(test_r2),
                )

                # Log metrics to MLflow
                mlflow.log_metrics(
                    {
                        "mean_test_loss": np.mean(test_loss),
                        "mean_test_mape": np.mean(test_mape),
                        "mean_test_rmse": np.mean(test_rmse),
                        "mean_test_r2": np.mean(test_r2),
                    }
                )

                # Return the metric we want to optimize
                if self.optimize_metric.lower() == "rmse":
                    return np.mean(test_rmse)
                elif self.optimize_metric.lower() == "mape":
                    return np.mean(test_mape)
                elif self.optimize_metric.lower() == "loss":
                    return np.mean(test_loss)
                elif self.optimize_metric.lower() == "r2":
                    return -np.mean(
                        test_r2
                    )  # Negative because Optuna minimizes by default
                else:
                    raise ValueError(
                        f"Unknown optimization metric: {self.optimize_metric}"
                    )

            except Exception as e:
                logging.error("Error in trial %d: %s", trial.number, str(e))
                mlflow.log_param("error", str(e))
                raise optuna.exceptions.TrialPruned()

    def update_config_file(self, config: Dict):
        """
        Overwrite the existing config file with the updated hyperparameters.
        """
        try:
            # Load the existing config file
            with open(self.config_path, "r", encoding="utf8") as f:
                full_config = json.load(f)

            # Update the relevant parts of the config
            full_config["kwargs"].update(config)

            # Save the updated config
            with open(self.config_path, "w", encoding="utf8") as f:
                json.dump(full_config, f, indent=4)
        except Exception as e:
            logging.error("Error updating config file: %s", str(e))
            raise

    def extract_average_performance(
        self, test_metrics_list: List[Tuple[str, np.ndarray, np.ndarray, Dict]]
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Returns:
            test_loss (list): Each model's average test loss.
            test_mape (list): Each model's average test MAPE.
            test_rmse (list): Each model's average test RMSE.
            test_r2 (list): Each model's average test R2.
        """
        # Extract the precalculated average performance metrics and sensor name from the test_metrics_list
        avg_performance = [tuple[3] for tuple in test_metrics_list]
        test_loss = [dict["Test loss"] for dict in avg_performance]
        test_mape = [dict["Test MAPE"] for dict in avg_performance]
        test_rmse = [dict["Test RMSE"] for dict in avg_performance]
        test_r2 = [dict["Test R2"] for dict in avg_performance]

        return test_loss, test_mape, test_rmse, test_r2

    def run_tuning(self, n_trials: int = 100) -> Dict:
        """
        Run the hyperparameter tuning process using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(
            self.objective, n_trials=n_trials, callbacks=[self.print_best_trial]
        )

        logging.info("Best trial:")
        trial = study.best_trial
        logging.info("  Value: %s", trial.value)
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info("    %s: %s", key, value)

        # Save the best hyperparameters and study statistics to a JSON file
        self.save_results(study)

        return study.best_params

    def save_results(self, study: optuna.study.Study):
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "optimization_history": [
                {"trial_number": t.number, "value": t.value} for t in study.trials
            ],
        }

        with open(self.results_file, "w", encoding="utf8") as f:
            json.dump(results, f, indent=4)

        logging.info("Results saved to: %s", self.results_file)

    @staticmethod
    def print_best_trial(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """
        Callback function to print the best trial's results.
        """
        if study.best_trial.number == trial.number:
            logging.info("New best trial: %d", trial.number)
            logging.info("  Value: %s", trial.value)
            logging.info("  Params: ")
            for key, value in trial.params.items():
                logging.info("    %s: %s", key, value)


# Usage
if __name__ == "__main__":
    tuner = HyperparameterTuner(PIPELINE_CONFIG_PATH, optimize_metric="r2")
    best_params = tuner.run_tuning()
