from datetime import date
import pickle
import pandas as pd

from src.utils.general_utils import load_config
from src.utils.polygon_utils import create_wkb_polygon
from src.utils.general_utils import (
    get_window_size_from_config,
    get_horizon_from_config,
    get_stride_from_config,
    get_batch_size_from_config,
    get_model_type_from_config,
)

from src.utils.pipeline_types import (
    RawDataItem,
    PreprocessedItem,
    EngineeredItem,
    DataLoaderItem,
    TrainedModelItem,
    EvaluationItem,
)


def create_raw_file_path_from_config() -> str:
    """
    Creates a file path for storing or retrieving sensor data based on the API configuration.

    Returns:
        str: The file path for storing or retrieving the sensor data.
    """
    api_config_path = "configs/api_config.json"
    api_config = load_config(api_config_path)
    today = date.today()
    last_n_days = api_config["api"]["endpoints"]["raw_sensor_data"]["params"][
        "last_n_days"
    ]
    coords = api_config["api"]["coords"]
    bbox = create_wkb_polygon(coords[0], coords[1], coords[2], coords[3])
    bbox = bbox[-16:]
    file_path = f"data/raw/{today}_Last_{last_n_days}_Days_{bbox}.pkl"
    return file_path


def create_dataloaders_file_path_from_config() -> str:
    """
    Creates a processed file path based on the configuration settings.

    The processed file path is constructed by combining the raw file path (obtained from the
    `create_raw_file_path_from_config()` function) and the values of `window_size`, `horizon`,
    `stride`, and `batch_size` retrieved from the configuration file.

    The `.pkl` extension is removed from the raw file path before constructing the processed file path.

    The format of the processed file path is:
    "{raw_file_path}_WindowSize_{window_size}_Horizon_{horizon}_Stride_{stride}_BatchSize_{batch_size}.pkl"

    Returns:
        str: The constructed processed file path.

    Example:
        If the raw file path is "data/raw/sensor_data.pkl" and the configuration settings are:
        - window_size: 10
        - horizon: 5
        - stride: 3
        - batch_size: 32

        The resulting processed file path would be:
        "data/processed/training_data/sensor_data_WindowSize_10_Horizon_5_Stride_3_BatchSize_32.pkl"
    """
    raw_file_path = (
        create_raw_file_path_from_config().rstrip(".pkl").replace("raw", "dataloaders")
    )
    window_size = get_window_size_from_config()
    horizon = get_horizon_from_config()
    stride = get_stride_from_config()
    batch_size = get_batch_size_from_config()
    model_type = get_model_type_from_config()

    processed_file_path = f"{raw_file_path}_WindowSize_{window_size}_Horizon_{horizon}_Stride_{stride}_BatchSize_{batch_size}_Model_{model_type}.pkl"

    return processed_file_path


def load_data(file_path, data_type):
    """
    Fetches data from local storage if available, or downloads it if not.

    Args:
        file_path (str): The path to the data file.
        data_type (str): The type of data being loaded (e.g., "raw", "preprocessed", "engineered", "dataloaders").

    Returns:
        The loaded data.
    """

    print(f"\nReading in {data_type} data from local storage...\n")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def load_raw_data() -> RawDataItem:
    file_path = create_raw_file_path_from_config()
    return load_data(file_path, "raw")


def load_preprocessed_data() -> PreprocessedItem:
    file_path = create_raw_file_path_from_config().replace("raw", "preprocessed")
    return load_data(file_path, "preprocessed")


def load_engineered_data() -> EngineeredItem:
    file_path = create_raw_file_path_from_config().replace("raw", "engineered")
    return load_data(file_path, "engineered")


def load_dataloaders() -> DataLoaderItem:
    file_path = create_dataloaders_file_path_from_config()
    return load_data(file_path, "dataloaders")


def load_trained_models() -> TrainedModelItem:
    file_path = create_dataloaders_file_path_from_config().replace(
        "dataloaders", "trained_models"
    )
    return load_data(file_path, "trained_models")


def load_evaluation_metrics() -> EvaluationItem:
    file_path = create_dataloaders_file_path_from_config().replace(
        "dataloaders", "evaluation"
    )
    return load_data(file_path, "evaluation")


def save_data(data, file_path, data_type):
    """
    Saves data to local storage.

    Args:
        data: The data to be saved.
        file_path (str): The path to save the data file.
        data_type (str): The type of data being saved (e.g., "raw", "preprocessed", "engineered", "dataloaders").
    """
    print(f"\nSaving {data_type} data to local storage...\n")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def save_raw_data(data, file_path):
    save_data(data, file_path, "raw")


def save_preprocessed_data(data, file_path):
    save_data(data, file_path, "preprocessed")


def save_engineered_data(data, file_path):
    save_data(data, file_path, "engineered")


def save_dataloaders(data, file_path):
    save_data(data, file_path, "dataloaders")


def save_trained_models(data, file_path):
    save_data(data, file_path, "trained models")


def save_evaluation_metrics(data, file_path):
    save_data(data, file_path, "evaluation metrics")
