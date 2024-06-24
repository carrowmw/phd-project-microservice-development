import json
import re


# Load configuration from JSON
def load_config(config_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): The file path to the configuration JSON file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
        return config


def get_step_config(step_name, config):
    """
    Retrieves the configuration for a specific preprocessing step.

    Args:
        step_name (str): The name of the preprocessing step.
        config (dict): The full configuration dictionary.

    Returns:
        dict: The configuration dictionary for the specified step. Returns an empty dictionary if the step is not found.
    """
    for step in config.get("preprocessing_steps", []):
        if step["name"] == step_name:
            return step.get("kwargs", {})
    return {}


def should_execute_step(step_config):
    """
    Determines whether a step should be executed based on its configuration.

    Args:
        step_config (dict): The configuration for the step, containing optional execute_step boolean.

    Returns:
        bool: True if the step should be executed, False otherwise.
    """
    # Default to True if execute_step is not specified
    return step_config.get("execute_step", True)


def extract_values_from_filename(filename):
    """
    Extracts the Completeness, Sequence Length, Horizon, and Window Size values
    from the given filename using regular expressions.

    Args:
    - filename (str): The filename to extract the values from.

    Returns:
    - dict: A dictionary containing the extracted values. Returns None for values not found.
    """
    # Regular expressions for each value
    regex_patterns = {
        "Completeness": r"Completeness([\d.]+)",
        "SequenceLength": r"SequenceLength(\d+)",
        "Horizon": r"Horizon(\d+)",
        "WindowSize": r"WindowSize(\d+)",
        "TestNumber": r"TestNumber(\d+)",
    }

    extracted_values = {}
    for key, pattern in regex_patterns.items():
        match = re.search(pattern, filename)
        if match:
            # Convert to float if it has a decimal point, else convert to int
            value = (
                float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            )
            extracted_values[key] = value
        else:
            extracted_values[key] = None

    return extracted_values


def get_config_step_value(config_file, step_name, param_name):
    """
    Retrieves a specific parameter value from the specified config file.

    Args:
        config_file (str): Path to the config file.
        step_name (str): Name of the step in the config file.
        param_name (str): Name of the parameter to retrieve.

    Returns:
        The value of the specified parameter.

    Raises:
        ValueError: If the specified parameter is not found in the config file.
    """
    config = load_config(config_file)
    for step in config["steps"]:
        if step["name"] == step_name:
            if param_name in step["kwargs"]:
                return step["kwargs"][param_name]
    raise ValueError(f"Could not find {param_name} in the provided config file.")


def get_config_kwarg_value(config_file, kwarg_name):
    """
    Retrieves a specific parameter value from the specified config file.

    Args:
        config_file (str): Path to the config file.
        kwarg_name (str): Name of the kwarg to retrieve.

    Returns:
        The value of the specified kwarg.

    Raises:
        ValueError: If the specified kwarg is not found in the config file.
    """
    config = load_config(config_file)
    if kwarg_name in config["kwargs"]:
        return config["kwargs"][kwarg_name]
    raise ValueError(f"Could not find {kwarg_name} in the provided config file.")


def get_model_type_from_config():
    """
    Retrieves the window_size value from the dataloader_config.json file.

    Args:
        config_file (str): Path to the dataloader_config.json file.

    Returns:
        int: The value of window_size specified in the config file.
    """
    config_file = "phd_project/configs/dataloader_config.json"
    config = load_config(config_file)
    model_type = config["kwargs"]["model_type"]
    if model_type is None:
        raise ValueError("Could not find model_type in the provided config file.")
    return model_type


def get_window_size_from_config():
    """
    Retrieves the window_size value from the dataloader_config.json file.

    Returns:
        int: The value of window_size specified in the config file.

    Raises:
        ValueError: If the window_size parameter is not found in the config file.
    """
    return get_config_step_value(
        "phd_project/configs/dataloader_config.json",
        "phd_project.src.training.dataloader.sliding_windows",
        "window_size",
    )


def get_horizon_from_config():
    """
    Retrieves the horizon value from the dataloader_config.json file.

    Returns:
        int: The value of horizon specified in the config file.

    Raises:
        ValueError: If the horizon parameter is not found in the config file.
    """
    return get_config_step_value(
        "phd_project/configs/dataloader_config.json",
        "phd_project.src.training.dataloader.sliding_windows",
        "horizon",
    )


def get_stride_from_config():
    """
    Retrieves the stride value from the dataloader_config.json file.

    Returns:
        int: The value of stride specified in the config file.

    Raises:
        ValueError: If the stride parameter is not found in the config file.
    """
    return get_config_step_value(
        "phd_project/configs/dataloader_config.json",
        "phd_project.src.training.dataloader.sliding_windows",
        "stride",
    )


def get_batch_size_from_config():
    """
    Retrieves the batch_size value from the dataloader_config.json file.

    Returns:
        int: The value of batch_size specified in the config file.

    Raises:
        ValueError: If the batch_size parameter is not found in the config file.
    """
    return get_config_step_value(
        "phd_project/configs/dataloader_config.json",
        "phd_project.src.training.dataloader.create_dataloaders",
        "batch_size",
    )


def get_aggregation_frequency_mins_from_config():
    """
    Retrieves the aggregation frequency in minutes from the preprocessing_config.json file.

    Returns:
        int: The value of aggregation_frequency_mins specified in the config file.

    Raises:
        ValueError: If the aggregation_frequency_mins parameter is not found in the config file.
    """
    return get_config_step_value(
        "phd_project/configs/preprocessing_config.json",
        "phd_project.src.data_processing.preprocessing.aggregate_on_datetime",
        "aggregation_frequency_mins",
    )


def get_datetime_column_from_config():
    """
    Retrieves the datetime_column value from the preprocessing_config.json file.

    Returns:
        str: The value of datetime_column specified in the config file.

    Raises:
        ValueError: If the datetime_column parameter is not found in the config file.
    """
    return get_config_kwarg_value(
        "phd_project/configs/preprocessing_config.json", "datetime_column"
    )


def get_value_column_from_config():
    """
    Retrieves the value_column value from the preprocessing_config.json file.

    Returns:
        str: The value of value_column specified in the config file.

    Raises:
        ValueError: If the value_column parameter is not found in the config file.
    """
    return get_config_kwarg_value(
        "phd_project/configs/preprocessing_config.json", "value_column"
    )


def get_completeness_threshold_from_config():
    """
    Retrieves the completeness threshold from the api_config.json file.

    Returns:
        float: The value of completeness specified in the config file.

    Raises:
        ValueError: If the completeness parameter is not found in the config file.
    """
    return get_config_kwarg_value(
        "phd_project/configs/preprocessing_config.json", "completeness_threshold"
    )


def get_last_n_days_from_config():
    """
    Retrieves the last_n_days value from the api_config.json file.

    Returns:
        int: The value of last_n_days specified in the config file.

    Raises:
        ValueError: If the last_n_days parameter is not found in the config file.
    """
    config_file = "phd_project/configs/api_config.json"
    with open(config_file, encoding="utf-8") as file:
        config = json.load(file)

    raw_sensor_data_endpoint = config["api"]["endpoints"]["raw_sensor_data"]
    last_n_days = raw_sensor_data_endpoint.get("params", {}).get("last_n_days")

    return last_n_days
