"""
This module handles the packing and unpacking of the datalists that are passed along the pipeline.
"""

from src.utils.pipeline_utils import process_data, load_or_process_data
from src.data_processing.execute_requests import (
    execute_raw_sensor_data_request,
    execute_sensors_request,
    print_sensor_request_metrics,
)
from src.utils.pipeline_types import (
    check_type,
    RawDataItem,
    PreprocessedItem,
    EngineeredItem,
    DataLoaderItem,
    TrainedModelItem,
    EvaluationItem,
)

from src.utils.general_utils import get_window_size_from_config

from src.utils.fstore_utils import (
    create_raw_file_path_from_config,
    create_dataloaders_file_path_from_config,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
    load_trained_models,
    load_evaluation_metrics,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloaders,
    save_trained_models,
    save_evaluation_metrics,
)


def download_raw_data() -> RawDataItem:
    """
    Downloads raw sensor data from the API.

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
    preprocessing_config_path = "configs/preprocessing_config.json"

    def process_preprocessed_data(dfs):
        print(f"\n\nPreprocessing {len(dfs)} DataFrames...\n")
        preprocessed_dfs = []
        for i, df in enumerate(dfs, start=1):
            print(f"\nProcessing DataFrame {i}/{len(dfs)}")
            preprocessed_df = process_data(df[1], preprocessing_config_path)
            preprocessed_dfs.append((df[0], preprocessed_df))
        return preprocessed_dfs

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
    feature_engineering_config_path = "configs/feature_engineering_config.json"

    def process_engineered_data(dfs):
        print(f"\n\nApplying feature engineering to {len(dfs)} DataFrames...\n")
        engineered_dfs = []
        for i, df in enumerate(dfs, start=1):
            if not df[1].empty:
                print(f"\nProcessing DataFrame {i}/{len(dfs)}")
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
    dataloader_config = "configs/dataloader_config.json"
    window_size = get_window_size_from_config()

    def process_dataloaders(dfs):
        print(f"\n\nLoading data for {len(dfs)} DataFrames...\n")
        list_of_dataloaders = []
        for i, df in enumerate(dfs, start=1):
            if len(df[1]) > window_size * 10:
                print(f"\nProcessing DataFrame {i}/{len(dfs)}")
                model, train_loader, val_loader, test_loader = process_data(
                    df[1], dataloader_config
                )
                list_of_dataloaders.append(
                    (df[0], model, train_loader, val_loader, test_loader)
                )
        return list_of_dataloaders

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
    training_config_path = "configs/training_config.json"

    def process_trained_models(loaders):
        print(f"\n\nTraining {len(loaders)} models...\n")
        list_of_trained_models_and_metrics = []
        for i, loader in enumerate(loaders, start=1):
            print(f"\nTraining model {i}/{len(loaders)}")
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
    evaluation_config_path = "configs/evaluation_config.json"

    def process_evaluation_metrics(models):
        print(f"\n\nEvaluating {len(models)} models...\n")
        list_of_evaluation_metrics = []
        for i, model in enumerate(models, start=1):
            print(f"\nEvaluating model {i}/{len(models)}")
            test_predictions, test_labels, test_metrics = process_data(
                (model[1], model[2]), evaluation_config_path
            )
            list_of_evaluation_metrics.append(
                (model[0], test_predictions, test_labels, test_metrics)
            )
        return list_of_evaluation_metrics

    evaluation_metrics_list = process_evaluation_metrics(trained_models_list)

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
