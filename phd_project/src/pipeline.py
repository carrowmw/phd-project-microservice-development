"""
This module handles the packing and unpacking of the datalists that are passed along the pipeline.
"""

import sys
import os

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure the src directory is in the Python path
src_path = os.path.join(project_root, "phd_project", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import logging
import pandas as pd
from phd_project.src.utils.pipeline_utils import process_data, load_or_process_data
from phd_project.src.data_processing.execute_requests import (
    execute_raw_sensor_data_request,
    execute_sensors_request,
    print_sensor_request_metrics,
)
from phd_project.src.utils.pipeline_types import (
    check_type,
    SensorListItem,
    RawDataItem,
    PreprocessedItem,
    EngineeredItem,
    DataLoaderItem,
    TrainedModelItem,
    EvaluationItem,
)

from phd_project.src.utils.general_utils import get_window_size_from_config

from phd_project.src.utils.fstore_utils import (
    create_raw_file_path_from_config,
    create_dataloaders_file_path_from_config,
    load_sensor_list,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
    load_trained_models,
    load_evaluation_metrics,
    save_sensor_list,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloaders,
    save_trained_models,
    save_evaluation_metrics,
)


def download_sensor_list() -> SensorListItem:
    """
    Downloads the list of sensors from the API.
    """

    def process_sensor_list():
        sensors_df = execute_sensors_request()
        return sensors_df

    sensors_df = load_or_process_data(
        None,
        create_raw_file_path_from_config().replace("raw", "sensors"),
        load_sensor_list,
        process_sensor_list,
        save_sensor_list,
        "sensor list download",
    )

    assert isinstance(
        sensors_df, pd.DataFrame
    ), f"Expected sensors_df to be a pd.DataFrame, but got {type(sensors_df)}"

    return sensors_df


def read_or_download_raw_data() -> RawDataItem:
    """
    Reads in from local storage if available or downloads raw sensor data from the API.

    Returns:
        List[pd.DataFrame]: The raw sensor data.
    """

    def process_raw_data():
        sensors_df = execute_sensors_request()
        series_of_sensor_names = sensors_df["Sensor Name"]
        raw_dfs = execute_raw_sensor_data_request(sensors_df)
        print_sensor_request_metrics(raw_dfs, series_of_sensor_names)
        return raw_dfs

    raw_dfs = load_or_process_data(
        None,
        create_raw_file_path_from_config(),
        load_raw_data,
        process_raw_data,
        save_raw_data,
        "raw data download",
    )

    assert isinstance(
        raw_dfs, list
    ), f"Expected raw_dfs to be a list, but got {type(raw_dfs)}"
    for item in raw_dfs:
        # pylint: disable=no-member
        check_type(item, RawDataItem.__args__[0])

    return raw_dfs


def preprocess_raw_data(raw_dfs) -> PreprocessedItem:
    """ """
    preprocessing_config_path = "phd_project/configs/preprocessing_config.json"

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
            preprocessed_df = process_data(df[1], preprocessing_config_path)
            preprocessed_dfs.append((df[0], preprocessed_df))
        return preprocessed_dfs

    print(create_raw_file_path_from_config().replace("raw", "preprocessed"))
    preprocessed_dfs = load_or_process_data(
        raw_dfs,
        create_raw_file_path_from_config().replace("raw", "preprocessed"),
        load_preprocessed_data,
        process_preprocessed_data,
        save_preprocessed_data,
        "preprocessing",
    )

    assert isinstance(
        preprocessed_dfs, list
    ), f"Expected preprocessed_dfs to be a list, but got {type(preprocessed_dfs)}"
    for item in preprocessed_dfs:
        # pylint: disable=no-member
        check_type(item, PreprocessedItem.__args__[0])

    return preprocessed_dfs


def apply_feature_engineering(preprocessed_dfs) -> EngineeredItem:
    """ """
    feature_engineering_config_path = (
        "phd_project/configs/feature_engineering_config.json"
    )

    def process_engineered_data(dfs):
        print(f"\n\nApplying feature engineering to {len(dfs)} DataFrames...\n")
        logging.info("\n\nApplying feature engineering to %d DataFrames...\n", len(dfs))
        engineered_dfs = []
        for i, df in enumerate(dfs, start=1):
            if not df[1].empty:
                print(f"\nEngineering DataFrame {i}/{len(dfs)} - {df[0]}")
                logging.info(
                    "\nEngineering DataFrame %d/%d - %s", (i + 1), len(dfs), df[0]
                )

                engineered_df = process_data(df[1], feature_engineering_config_path)
                engineered_dfs.append((df[0], engineered_df))

        return engineered_dfs

    engineered_dfs = load_or_process_data(
        preprocessed_dfs,
        create_raw_file_path_from_config().replace("raw", "engineered"),
        load_engineered_data,
        process_engineered_data,
        save_engineered_data,
        "feature engineering",
    )

    assert isinstance(
        engineered_dfs, list
    ), f"Expected engineered_dfs to be a list, but got {type(engineered_dfs)}"
    for item in engineered_dfs:
        # pylint: disable=no-member
        check_type(item, EngineeredItem.__args__[0])

    return engineered_dfs


def load_data(engineered_dfs) -> DataLoaderItem:
    dataloader_config = "phd_project/configs/dataloader_config.json"
    window_size = get_window_size_from_config()

    def process_dataloaders(dfs):
        print(f"\n\nLoading data for {len(dfs)} DataFrames...\n")
        logging.info("\n\nLoading data for %d DataFrames...\n", len(dfs))
        list_of_dataloaders = []
        for i, df in enumerate(dfs, start=1):
            if len(df[1]) > window_size * 10:
                print(f"\nProcessing DataLoader {i}/{len(dfs)} - {df[0]}")
                logging.info(
                    "\nProcessing DataLoader %d/%d - %s", (i + 1), len(dfs), df[0]
                )
                model, train_loader, val_loader, test_loader = process_data(
                    df[1], dataloader_config
                )
                list_of_dataloaders.append(
                    (df[0], model, train_loader, val_loader, test_loader)
                )
        return list_of_dataloaders

    print(create_dataloaders_file_path_from_config())
    list_of_dataloaders = load_or_process_data(
        engineered_dfs,
        create_dataloaders_file_path_from_config(),
        load_dataloaders,
        process_dataloaders,
        save_dataloaders,
        "data loading",
    )

    assert isinstance(
        list_of_dataloaders, list
    ), f"Expected list_of_dataloaders to be a list, but got {type(list_of_dataloaders)}"
    for item in list_of_dataloaders:
        # pylint: disable=no-member
        check_type(item, DataLoaderItem.__args__[0])

    return list_of_dataloaders


def train_model(dataloaders_list) -> TrainedModelItem:
    """ """
    training_config_path = "phd_project/configs/training_config.json"

    def process_trained_models(loaders):
        print(f"\n\nTraining {len(loaders)} models...\n")
        logging.info("\n\nTraining %d models...\n", len(loaders))
        list_of_trained_models_and_metrics = []
        for i, loader in enumerate(loaders, start=1):
            print(f"\nTraining model {i}/{len(loaders)} - {loader[0]}")
            logging.info(
                "\nTraining model %d/%d - %s", (i + 1), len(loaders), loader[0]
            )
            model, train_metrics, val_metrics = process_data(
                (loader[1], loader[2], loader[3]), training_config_path
            )
            list_of_trained_models_and_metrics.append(
                (loader[0], model, loader[4], train_metrics, val_metrics)
            )
        return list_of_trained_models_and_metrics

    trained_models_and_metrics_list = load_or_process_data(
        dataloaders_list,
        create_dataloaders_file_path_from_config().replace(
            "dataloaders", "trained_models"
        ),
        load_trained_models,
        process_trained_models,
        save_trained_models,
        "model training",
    )

    assert isinstance(
        trained_models_and_metrics_list, list
    ), f"Expected trained_models_and_metrics_list to be a list, but got {type(trained_models_and_metrics_list)}"
    for item in trained_models_and_metrics_list:
        # pylint: disable=no-member
        check_type(item, TrainedModelItem.__args__[0])

    return trained_models_and_metrics_list


def evaluate_model(trained_models_list) -> EvaluationItem:
    """"""
    evaluation_config_path = "phd_project/configs/evaluation_config.json"

    def process_evaluation_metrics(models):
        print(f"\n\nEvaluating {len(models)} models...\n")
        logging.info("\n\nEvaluating %d models...\n", len(models))
        list_of_evaluation_metrics = []
        for i, model in enumerate(models, start=1):
            print(f"\nEvaluating model {i}/{len(models)} - {model[0]}")
            logging.info(
                "\nEvaluating model %d/%d - %s", (i + 1), len(models), model[0]
            )
            test_predictions, test_labels, test_metrics = process_data(
                (model[1], model[2]), evaluation_config_path
            )
            list_of_evaluation_metrics.append(
                (model[0], test_predictions, test_labels, test_metrics)
            )
        return list_of_evaluation_metrics

    evaluation_metrics_list = load_or_process_data(
        trained_models_list,
        create_dataloaders_file_path_from_config().replace("dataloaders", "evaluation"),
        load_evaluation_metrics,
        process_evaluation_metrics,
        save_evaluation_metrics,
        "model evaluation",
    )

    assert isinstance(
        evaluation_metrics_list, list
    ), f"Expected evaluation_metrics_list to be a list, but got {type(evaluation_metrics_list)}"
    for item in evaluation_metrics_list:
        # pylint: disable=no-member
        check_type(item, EvaluationItem.__args__[0])

    return evaluation_metrics_list
