from typing import TypeAlias, Tuple, List, get_args, Callable
import os
import pickle
from datetime import date
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from config.paths import (
    get_api_config_path,
    get_sensor_dir,
    get_raw_data_dir,
    get_preprocessed_data_dir,
    get_engineered_data_dir,
    get_dataloader_dir,
    get_trained_models_dir,
    get_evaluation_dir,
)
from utils.config_helper import (
    get_polygon_wkb,
    get_last_n_days,
    load_config,
    get_window_size,
    get_horizon,
    get_stride,
    get_batch_size,
    get_model_type,
    get_epochs,
)


def check_type(item, expected_type):
    assert isinstance(item, tuple), f"Expected item to be a tuple, but got {type(item)}"
    assert len(item) == len(
        get_args(expected_type)
    ), f"Expected item to have {len(get_args(expected_type))} elements, but got {len(item)}"
    for i, (elem, expected_elem_type) in enumerate(zip(item, get_args(expected_type))):
        assert isinstance(
            elem, expected_elem_type
        ), f"Expected element {i} to be of type {expected_elem_type}, but got {type(elem)}"


SensorListItem: TypeAlias = pd.DataFrame

# Type alias for the raw data stage
RawDataItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the preprocessing stage
PreprocessedItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the feature engineering stage
EngineeredItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the data loading stage
DataLoaderItem: TypeAlias = list[
    Tuple[str, nn.Module, DataLoader, DataLoader, DataLoader]
]

# Type alias for the model training stage
TrainedModelItem: TypeAlias = list[Tuple[str, nn.Module, DataLoader, List, List]]

# Type alias for the evaluation stage
EvaluationItem: TypeAlias = list[Tuple[str, np.ndarray, np.ndarray, dict]]


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


def save_data(data, file_path, data_type):
    """
    Saves data to local storage.

    Args:
        data: The data to be saved.
        file_path (str): The path to save the data file.
        data_type (str): The type of data being saved (e.g., "raw", "preprocessed", "engineered", "dataloaders").
    """
    print(f"\nSaving {data_type} data to local storage...\n")

    # Create directory if it does not exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def find_tuple_by_first_element(tuples, search_string, n_tuples=1):
    """
    Finds a tuple in a list of tuples by the first element of the tuple.

    Args:
        tuples (list): A list of tuples.
        search_string (str): The string to search for in the first element of the tuples.

    Returns:
        The tuple containing the search string if found, otherwise None.
    """
    for tuple_item in tuples:
        if tuple_item[0] == search_string:
            tupleitem = tuple_item[1 : n_tuples + 1]
            if len(tupleitem) == 1:
                return tupleitem[0]
            return tupleitem
    return None


def json_to_dataframe(json_data):
    """
    Convert a JSON object to a pandas DataFrame.

    Parameters:
    - json_data (dict): The JSON data to convert.

    Returns:
    - DataFrame: A pandas DataFrame constructed from the input JSON data.
    """
    return pd.DataFrame(json_data)


def pipeline_input_data_filename() -> str:
    """
    Creates a file path for storing or retrieving sensor data based on the API configuration.

    Returns:
        str: The file path for storing or retrieving the sensor data.
    """
    today = date.today()
    last_n_days = get_last_n_days()
    bbox = get_polygon_wkb()
    bbox = bbox[-8:]
    file_path = f"{today}_Last_{last_n_days}_Days_{bbox}.pkl"
    return file_path


def pipeline_output_data_filename() -> str:
    """
    Creates a file path for storing or retrieving sensor data based on the API configuration.

    Returns:
        str: The file path for storing or retrieving the sensor data.
    """
    prefix = pipeline_input_data_filename().rstrip(".pkl")
    window_size = get_window_size()
    horizon = get_horizon()
    stride = get_stride()
    batch_size = get_batch_size()
    model_type = get_model_type()
    epochs = get_epochs()

    file_path = f"{prefix}_WindowSize_{window_size}_Horizon_{horizon}_Stride_{stride}_BatchSize_{batch_size}_Model_{model_type}_Epoch_{epochs}.pkl"

    return file_path


def create_file_path(data_path_func: Callable, file_name_func: Callable):
    """
    Creates a file path for storing or retrieving data.

    Args:
        data_path_func (Callable): The function that returns the data path.
        file_name_func (Callable): The function that returns the file name.

    Returns:
        str: The file path for storing or retrieving the data.
    """
    data_path = data_path_func()
    file_name = file_name_func()
    return f"{data_path}/{file_name}"


def load_sensor_list() -> SensorListItem:
    file_path = create_file_path(get_sensor_dir, pipeline_input_data_filename)
    return load_data(file_path, "sensors")


def load_raw_data() -> RawDataItem:
    file_path = create_file_path(get_raw_data_dir, pipeline_input_data_filename)
    return load_data(file_path, "raw")


def load_preprocessed_data() -> PreprocessedItem:
    file_path = create_file_path(
        get_preprocessed_data_dir, pipeline_input_data_filename
    )
    return load_data(file_path, "preprocessed")


def load_engineered_data() -> EngineeredItem:
    file_path = create_file_path(get_engineered_data_dir, pipeline_input_data_filename)
    return load_data(file_path, "engineered")


def load_dataloaders() -> DataLoaderItem:
    file_path = create_file_path(get_dataloader_dir, pipeline_output_data_filename)
    return load_data(file_path, "dataloaders")


def load_trained_models() -> TrainedModelItem:
    file_path = create_file_path(get_trained_models_dir, pipeline_output_data_filename)
    return load_data(file_path, "trained models")


def load_evaluation_metrics() -> EvaluationItem:
    file_path = create_file_path(get_evaluation_dir, pipeline_output_data_filename)
    return load_data(file_path, "evaluation metrics")


def save_sensor_list(data, file_path):
    save_data(data, file_path, "sensors")


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
    save_data(data, file_path, "evaluation_metrics")
