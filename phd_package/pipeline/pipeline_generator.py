# phd_package/pipeline/pipeline_generator.py

import os
import sys
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from ..config.paths import (
    get_pipeline_config_path,
    get_sensor_dir,
    get_raw_data_dir,
    get_preprocessed_data_dir,
    get_engineered_data_dir,
    get_dataloader_dir,
    get_trained_models_dir,
    get_test_dir,
)
from ..api.api_data_processor import APIDataProcessor

# from experiments.tracker import ExperimentTracker
from ..utils.pipeline_helper import (
    process_data,
    load_or_process_data,
    generate_random_string,
)
from ..utils.data_helper import (
    check_type,
    SensorListItem,
    RawDataItem,
    PreprocessedItem,
    EngineeredItem,
    DataLoaderItem,
    TrainedModelItem,
    TestItem,
    pipeline_input_data_filename,
    pipeline_processed_data_filename,
    pipeline_output_data_filename,
    create_file_path,
    load_sensor_list,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloader,
    load_trained_models,
    load_test_metrics,
    save_sensor_list,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloader,
    save_trained_models,
    save_test_metrics,
)
from ..utils.config_helper import get_window_size, get_horizon
from .utils.training_helper import check_mps_availability
from .conformal_prediction import ConformalPredictor, prepare_calibration_data


class Pipeline:

    def __init__(self, experiment_id: str = None, trial_number: int = None):
        # Initialize Experiment Tracker
        self.experiment_id = experiment_id or generate_random_string(12)
        self.trial_number = trial_number or 0
        self.get_or_create_experiment()

        self.train_dataloader = None
        self.model = None
        self.test_dataloader = None

        # self.experiment_tracker = ExperimentTracker(experiment_name)
        # Initialize logging
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get_or_create_experiment(self):
        experiment_name = f"Pipeline_Experiment_{self.experiment_id}"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logging.error("Error creating/getting experiment: %s", str(e))
            self.experiment_id = None

    def load_sensors(self) -> SensorListItem:
        """
        Downloads the list of sensors from the API.
        """

        def process_sensor_list():
            processor = APIDataProcessor()
            sensors_df = processor.execute_sensors_request()
            return sensors_df

        sensors_df = load_or_process_data(
            None,
            create_file_path(get_sensor_dir, pipeline_input_data_filename),
            load_sensor_list,
            process_sensor_list,
            save_sensor_list,
            "sensor list download",
        )

        assert isinstance(
            sensors_df, pd.DataFrame
        ), f"Expected sensors_df to be a pd.DataFrame, but got {type(sensors_df)}"

        return sensors_df

    def load_raw_data(self) -> RawDataItem:
        """
        Reads in from local storage if available or downloads raw sensor data from the API.
        """

        def process_raw_data():
            processor = APIDataProcessor()
            raw_dfs = processor.execute_data_request()
            return raw_dfs

        raw_dfs = load_or_process_data(
            None,
            create_file_path(get_raw_data_dir, pipeline_input_data_filename),
            load_raw_data,
            process_raw_data,
            save_raw_data,
            "raw data download",
        )

        assert isinstance(
            raw_dfs, list
        ), f"Expected raw_dfs to be a list, but got {type(raw_dfs)}"
        for item in raw_dfs:
            check_type(item, RawDataItem.__args__[0])

        return raw_dfs

    def load_preprocessed_data(self, raw_dfs) -> PreprocessedItem:
        """
        Preprocess the raw data.
        """

        def process_preprocessed_data(dfs):
            print(f"\n\nPreprocessing {len(dfs)} DataFrames...\n")
            logging.info("\n\nPreprocessing %d DataFrames...\n", len(dfs))
            preprocessed_dfs = []
            for i, df in enumerate(dfs, start=1):
                print(f"\nPreprocessing DataFrame {i}/{len(dfs)} - {df[0]}")
                if df[1].empty:
                    print(f"DataFrame {df[0]} is empty. Skipping preprocessing.")
                    continue
                logging.info(
                    "\nPreprocessing DataFrame %d/%d - %s", (i + 1), len(dfs), df[0]
                )
                preprocessed_df = process_data(
                    df[1], stage="preprocessing", config_path=get_pipeline_config_path()
                )
                preprocessed_dfs.append((df[0], preprocessed_df))
            return preprocessed_dfs

        preprocessed_dfs = load_or_process_data(
            raw_dfs,
            create_file_path(
                get_preprocessed_data_dir, pipeline_processed_data_filename
            ),
            load_preprocessed_data,
            process_preprocessed_data,
            save_preprocessed_data,
            "preprocessing",
        )

        assert isinstance(
            preprocessed_dfs, list
        ), f"Expected preprocessed_dfs to be a list, but got {type(preprocessed_dfs)}"
        for item in preprocessed_dfs:
            check_type(item, PreprocessedItem.__args__[0])

        return preprocessed_dfs

    def load_engineered_data(self, preprocessed_dfs) -> EngineeredItem:
        """
        Apply feature engineering to preprocessed data.
        """

        def process_engineered_data(dfs):
            print(f"\n\nApplying feature engineering to {len(dfs)} DataFrames...\n")
            logging.info(
                "\n\nApplying feature engineering to %d DataFrames...\n", len(dfs)
            )
            engineered_dfs = []
            for i, df in enumerate(dfs, start=1):
                if not df[1].empty:
                    print(f"\nEngineering DataFrame {i}/{len(dfs)} - {df[0]}")
                    logging.info(
                        "\nEngineering DataFrame %d/%d - %s", (i + 1), len(dfs), df[0]
                    )
                    engineered_df = process_data(
                        df[1],
                        stage="feature_engineering",
                        config_path=get_pipeline_config_path(),
                    )
                    engineered_dfs.append((df[0], engineered_df))
            return engineered_dfs

        engineered_dfs = load_or_process_data(
            preprocessed_dfs,
            create_file_path(get_engineered_data_dir, pipeline_processed_data_filename),
            load_engineered_data,
            process_engineered_data,
            save_engineered_data,
            "feature engineering",
        )

        assert isinstance(
            engineered_dfs, list
        ), f"Expected engineered_dfs to be a list, but got {type(engineered_dfs)}"
        for item in engineered_dfs:
            check_type(item, EngineeredItem.__args__[0])

        return engineered_dfs

    def load_dataloader(self, engineered_dfs) -> DataLoaderItem:
        """
        Load data into dataloaders.
        """
        window_size = get_window_size()
        horizon = get_horizon()

        def process_dataloader(dfs):
            print(f"\n\nLoading data for {len(dfs)} DataFrames...\n")
            logging.info("\n\nLoading data for %d DataFrames...\n", len(dfs))
            print(
                f"Generating sliding windows with window size {window_size} and horizon {horizon}..."
            )
            logging.info(
                f"Generating sliding windows with window size {window_size} and horizon {horizon}..."
            )
            list_of_dataloaders = []
            for i, df in enumerate(dfs, start=1):
                if len(df[1]) > window_size * 10:
                    print(f"\nProcessing DataLoader {i}/{len(dfs)} - {df[0]}")
                    logging.info(
                        "\nProcessing DataLoader %d/%d - %s", (i + 1), len(dfs), df[0]
                    )
                    model, train_loader, val_loader, test_loader = process_data(
                        df[1],
                        stage="dataloader",
                        config_path=get_pipeline_config_path(),
                    )
                    list_of_dataloaders.append(
                        (df[0], model, train_loader, val_loader, test_loader)
                    )
                else:
                    print("DataFrame is too small to create a DataLoader. Skipping.")
            return list_of_dataloaders

        list_of_dataloaders = load_or_process_data(
            engineered_dfs,
            create_file_path(get_dataloader_dir, pipeline_output_data_filename),
            load_dataloader,
            process_dataloader,
            save_dataloader,
            "data loading",
        )

        assert isinstance(
            list_of_dataloaders, list
        ), f"Expected list_of_dataloaders to be a list, but got {type(list_of_dataloaders)}"
        for item in list_of_dataloaders:
            check_type(item, DataLoaderItem.__args__[0])

        return list_of_dataloaders

    def load_trained_models(self, dataloaders_list) -> TrainedModelItem:
        """
        Train models.
        """

        def process_trained_models(loaders):
            print(f"\n\nTraining {len(loaders)} models...\n")
            logging.info("\n\nTraining %d models...\n", len(loaders))
            list_of_trained_models_and_metrics = []

            for i, loader in enumerate(loaders, start=1):
                model_name, model, train_loader, val_loader, test_loader = loader
                with mlflow.start_run(run_name=f"Train_{model_name}", nested=True):
                    print(f"\nTraining model {i}/{len(loaders)} - {model_name}")
                    logging.info(
                        "\nTraining model %d/%d - %s",
                        (i + 1),
                        len(loaders),
                        model_name,
                    )

                    model, train_metrics_list, val_metrics_list = process_data(
                        (model, train_loader, val_loader),
                        stage="training",
                        config_path=get_pipeline_config_path(),
                    )

                    # Log model specific metrics
                    mlflow.log_metrics(
                        {
                            f"{model_name}_train_loss": train_metrics_list[-1][
                                "Train loss"
                            ],
                            f"{model_name}_train_mape": train_metrics_list[-1][
                                "Train MAPE"
                            ],
                            f"{model_name}_train_rmse": train_metrics_list[-1][
                                "Train RMSE"
                            ],
                            f"{model_name}_val_loss": val_metrics_list[-1]["Val loss"],
                            f"{model_name}_val_mape": val_metrics_list[-1]["Val MAPE"],
                            f"{model_name}_val_rmse": val_metrics_list[-1]["Val RMSE"],
                            f"{model_name}_val_r2": val_metrics_list[-1]["Val R2"],
                        }
                    )

                    list_of_trained_models_and_metrics.append(
                        (
                            model_name,
                            model,
                            test_loader,
                            train_metrics_list,
                            val_metrics_list,
                        )
                    )

            return list_of_trained_models_and_metrics

        trained_models_and_metrics_list = load_or_process_data(
            dataloaders_list,
            create_file_path(get_trained_models_dir, pipeline_output_data_filename),
            load_trained_models,
            process_trained_models,
            save_trained_models,
            "load_trained_models",
        )

        assert isinstance(
            trained_models_and_metrics_list, list
        ), f"Expected trained_models_and_metrics_list to be a list, but got {type(trained_models_and_metrics_list)}"
        for item in trained_models_and_metrics_list:
            check_type(item, TrainedModelItem.__args__[0])

        return trained_models_and_metrics_list

    def load_test_metrics(self, trained_models_list) -> TestItem:
        """
        Test models.
        """

        def process_test_metrics(models):
            print(f"\n\nTesting {len(models)} models…\n")
            logging.info("\n\nTesting %d models…\n", len(models))
            list_of_test_metrics = []
            for i, model_data in enumerate(models, start=1):
                try:
                    if not isinstance(model_data, tuple) or len(model_data) != 5:
                        logging.error("Unexpected model structure: %s", model_data)
                        continue
                    model_name, model, test_loader = (
                        model_data[0],
                        model_data[1],
                        model_data[2],
                    )
                    print(f"\nTesting model {i}/{len(models)} - {model_name}")
                    logging.info(
                        "\nTesting model %d/%d - %s", (i + 1), len(models), model_name
                    )

                    with mlflow.start_run(run_name=f"Test_{model_name}", nested=True):
                        test_predictions, test_labels, test_metrics = process_data(
                            (model, test_loader),
                            stage="testing",
                            config_path=get_pipeline_config_path(),
                        )

                        # Log model specific metrics
                        mlflow.log_metrics(
                            {
                                f"{model_name}_test_loss": test_metrics["Test loss"],
                                f"{model_name}_test_mape": test_metrics["Test MAPE"],
                                f"{model_name}_test_rmse": test_metrics["Test RMSE"],
                                f"{model_name}_test_r2": test_metrics["Test R2"],
                            }
                        )

                    list_of_test_metrics.append(
                        (model_name, test_predictions, test_labels, test_metrics)
                    )

                except Exception as e:
                    logging.error("Error testing model %s: %s", model_name, str(e))
                    logging.error("Exception type: %s", type(e).__name__)

            return list_of_test_metrics

        test_metrics_list = load_or_process_data(
            trained_models_list,
            create_file_path(get_test_dir, pipeline_output_data_filename),
            load_test_metrics,
            process_test_metrics,
            save_test_metrics,
            "testing",
        )

        assert isinstance(
            test_metrics_list, list
        ), f"Expected test_metrics_list to be a list, but got {type(test_metrics_list)}"
        for item in test_metrics_list:
            check_type(item, TestItem.__args__[0])

        return test_metrics_list

    def load_conformal_predictions(self, trained_models_list) -> TestItem:
        """
        Generate conformal prediction intervals for the test predictions.
        """

        def process_conformal_predictions(models):
            print(f"\n\nGenerating conformal predictions for {len(models)} models...\n")
            logging.info(
                "\n\nGenerating conformal predictions for %d models...\n", len(models)
            )

            predictions_with_intervals = []
            for i, model_data in enumerate(models, start=1):
                try:
                    if not isinstance(model_data, tuple) or len(model_data) != 4:
                        logging.error("Unexpected model data structure: %s", model_data)
                        continue

                    model_name, predictions, labels, metrics = model_data
                    print(
                        f"\nProcessing conformal predictions for model {i}/{len(models)} - {model_name}"
                    )

                    with mlflow.start_run(
                        run_name=f"Conformal_{model_name}", nested=True
                    ):
                        # Get the corresponding model and dataloaders from trained_models_list
                        model_info = next(
                            m for m in trained_models_list if m[0] == model_name
                        )
                        model = model_info[1]
                        test_loader = model_info[2]

                        # Setup conformal predictor
                        device = check_mps_availability()
                        predictor = ConformalPredictor(significance_level=0.1)

                        # Get training data from the pipeline
                        new_train_loader, cal_loader = prepare_calibration_data(
                            self.train_dataloader, cal_ratio=0.2
                        )

                        # Calibrate and predict
                        predictor.calibrate(model, cal_loader, device)
                        pred, lower, upper = predictor.predict(
                            model, test_loader, device
                        )

                        # Calculate coverage and average interval width
                        coverage = np.mean((labels >= lower) & (labels <= upper))
                        interval_width = np.mean(upper - lower)

                        # Add conformal metrics to the existing metrics
                        metrics.update(
                            {
                                "conformal_coverage": coverage,
                                "avg_interval_width": interval_width,
                                "prediction_intervals": {
                                    "lower": lower,
                                    "upper": upper,
                                },
                            }
                        )

                        # Log metrics to MLflow
                        mlflow.log_metrics(
                            {
                                f"{model_name}_conformal_coverage": coverage,
                                f"{model_name}_avg_interval_width": interval_width,
                            }
                        )

                        predictions_with_intervals.append(
                            (model_name, predictions, labels, metrics)
                        )

                except Exception as e:
                    logging.error(
                        "Error in conformal prediction for model %s: %s",
                        model_name,
                        str(e),
                    )
                    logging.error("Exception type: %s", type(e).__name__)

            return predictions_with_intervals

        predictions_with_intervals = load_or_process_data(
            trained_models_list,
            create_file_path(get_test_dir, pipeline_output_data_filename),
            load_test_metrics,
            process_conformal_predictions,
            save_test_metrics,
            "conformal_prediction",
        )

        return predictions_with_intervals

    def run_pipeline(self):
        """
        Run the pipeline.
        """
        if self.experiment_id is None:
            logging.error("No experiment ID found. Exiting pipeline.")
            return
        try:
            with mlflow.start_run(
                run_name=f"Trial_{self.trial_number}_Pipeline",
                experiment_id=self.experiment_id,
                nested=True,
            ):
                sensors_df = self.load_sensors()
                raw_dfs = self.load_raw_data()
                preprocessed_dfs = self.load_preprocessed_data(raw_dfs)
                engineered_dfs = self.load_engineered_data(preprocessed_dfs)
                dataloaders_list = self.load_dataloader(engineered_dfs)
                trained_models_list = self.load_trained_models(dataloaders_list)
                test_metrics_list = self.load_test_metrics(trained_models_list)
                predictions_with_intervals = self.load_conformal_predictions(
                    test_metrics_list
                )

            return predictions_with_intervals
        except mlflow.exceptions.MlflowException as e:
            logging.error("MLflow error: %s", str(e))
            return self._run_pipeline_without_mlflow()

    def _run_pipeline_without_mlflow(self):
        logging.error("Running pipeline without MLflow tracking...")

        sensors_df = self.load_sensors()
        raw_dfs = self.load_raw_data()
        preprocessed_dfs = self.load_preprocessed_data(raw_dfs)
        engineered_dfs = self.load_engineered_data(preprocessed_dfs)
        dataloaders_list = self.load_dataloader(engineered_dfs)
        trained_models_list = self.load_trained_models(dataloaders_list)
        test_metrics_list = self.load_test_metrics(trained_models_list)
        predictions_with_intervals = self.load_conformal_predictions(test_metrics_list)

        return predictions_with_intervals
