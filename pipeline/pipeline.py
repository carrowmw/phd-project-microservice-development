import os
import sys
import logging
import pandas as pd
import mlflow
import mlflow.pytorch
from config.paths import (
    get_pipeline_config_path,
    get_sensor_dir,
    get_raw_data_dir,
    get_preprocessed_data_dir,
    get_engineered_data_dir,
    get_dataloader_dir,
    get_trained_models_dir,
    get_test_dir,
)
from api.api_data_processor import APIDataProcessor

# from experiments.tracker import ExperimentTracker
from utils.pipeline_helper import (
    process_data,
    load_or_process_data,
    generate_random_string,
)
from utils.data_helper import (
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
    load_dataloaders,
    load_trained_models,
    load_test_metrics,
    save_sensor_list,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloaders,
    save_trained_models,
    save_test_metrics,
)
from utils.config_helper import get_window_size, get_horizon


class Pipeline:

    def __init__(self, experiment_name: str = None):
        # Initialize Experiment Tracker
        if experiment_name is not None:
            self.experiment_name = experiment_name
        self.experiment_name = generate_random_string(12)
        # self.experiment_tracker = ExperimentTracker(experiment_name)
        # Initialize logging
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def read_or_download_sensors(self) -> SensorListItem:
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

    def read_or_download_data(self) -> RawDataItem:
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

    def preprocess_data(self, raw_dfs) -> PreprocessedItem:
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

    def apply_feature_engineering(self, preprocessed_dfs) -> EngineeredItem:
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

    def load_data(self, engineered_dfs) -> DataLoaderItem:
        """
        Load data into dataloaders.
        """
        window_size = get_window_size()
        horizon = get_horizon()

        def process_dataloaders(dfs):
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
            load_dataloaders,
            process_dataloaders,
            save_dataloaders,
            "data loading",
        )

        assert isinstance(
            list_of_dataloaders, list
        ), f"Expected list_of_dataloaders to be a list, but got {type(list_of_dataloaders)}"
        for item in list_of_dataloaders:
            check_type(item, DataLoaderItem.__args__[0])

        return list_of_dataloaders

    def train_model(self, dataloaders_list) -> TrainedModelItem:
        """
        Train models.
        """

        def process_trained_models(loaders):
            print(f"\n\nTraining {len(loaders)} models...\n")
            logging.info("\n\nTraining %d models...\n", len(loaders))
            list_of_trained_models_and_metrics = []

            experiment_id = mlflow.create_experiment(self.experiment_name)
            print(type(experiment_id))
            print(experiment_id)
            with mlflow.start_run(
                run_name="PARENT_RUN",
                experiment_id=experiment_id,
                tags={"version": "v1", "priority": "P1"},
                description="parent",
            ) as parent_run:

                for i, loader in enumerate(loaders, start=1):
                    with mlflow.start_run(
                        run_name=loader[0],
                        experiment_id=experiment_id,
                        nested=True,
                    ) as child_run:

                        print(f"\nTraining model {i}/{len(loaders)} - {loader[0]}")
                        logging.info(
                            "\nTraining model %d/%d - %s",
                            (i + 1),
                            len(loaders),
                            loader[0],
                        )

                        model, train_metrics, val_metrics = process_data(
                            (loader[1], loader[2], loader[3]),
                            stage="training",
                            config_path=get_pipeline_config_path(),
                        )

                        # print(f"Logging model {loader[0]} to MLflow...")
                        # mlflow.pytorch.log_model(model, "models")

                        # # Register the model
                        # model_name = f"sensor_model_{loader[0]}"
                        # model_uri = f"runs:/{child_run.info.run_id}/models/{loader[0]}"
                        # mlflow.register_model(model_uri, model_name)

                        list_of_trained_models_and_metrics.append(
                            (loader[0], model, loader[4], train_metrics, val_metrics)
                        )

                return list_of_trained_models_and_metrics

        trained_models_and_metrics_list = load_or_process_data(
            dataloaders_list,
            create_file_path(get_trained_models_dir, pipeline_output_data_filename),
            load_trained_models,
            process_trained_models,
            save_trained_models,
            "training",
        )

        assert isinstance(
            trained_models_and_metrics_list, list
        ), f"Expected trained_models_and_metrics_list to be a list, but got {type(trained_models_and_metrics_list)}"
        for item in trained_models_and_metrics_list:
            check_type(item, TrainedModelItem.__args__[0])

        return trained_models_and_metrics_list

    def test_model(self, trained_models_list) -> TestItem:
        """
        Test models.
        """

        def process_test_metrics(models):
            print(f"\n\nTesting {len(models)} models…\n")
            logging.info("\n\nTesting %d models…\n", len(models))
            list_of_test_metrics = []
            for i, model in enumerate(models, start=1):
                print(f"\nTesting model {i}/{len(models)} - {model[0]}")
                logging.info(
                    "\nTesting model %d/%d - %s", (i + 1), len(models), model[0]
                )
                test_predictions, test_labels, test_metrics = process_data(
                    (model[1], model[2]),
                    stage="testing",
                    config_path=get_pipeline_config_path(),
                )
                list_of_test_metrics.append(
                    (model[0], test_predictions, test_labels, test_metrics)
                )

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

    def run_pipeline(self):
        """
        Run the pipeline.
        """
        sensors_df = self.read_or_download_sensors()
        raw_dfs = self.read_or_download_data()
        preprocessed_dfs = self.preprocess_data(raw_dfs)
        engineered_dfs = self.apply_feature_engineering(preprocessed_dfs)
        dataloaders_list = self.load_data(engineered_dfs)
        trained_models_list = self.train_model(dataloaders_list)
        test_metrics_list = self.test_model(trained_models_list)
        return test_metrics_list
