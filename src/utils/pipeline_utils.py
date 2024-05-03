import os
import importlib
from functools import reduce
from src.utils.general_utils import load_config, should_execute_step


def load_or_process_data(
    data, file_path, load_func, process_func, save_func, step_name
):
    """
    Loads processed data from a file if it exists, otherwise processes the data and saves it to the file.

    Args:
        data: The input data to be processed. Can be None if not required by process_func.
        file_path (str): The path to the file where the processed data will be saved or loaded from.
        load_func (callable): A function that loads the processed data from the file.
        process_func (callable): A function that processes the input data.
        save_func (callable): A function that saves the processed data to the file.
        step_name (str): The name of the processing step (e.g., "preprocessing", "feature engineering").

    Returns:
        The processed data, either loaded from the file or obtained by processing the input data.
    """
    if os.path.exists(file_path):
        print(f"{step_name.capitalize()} data found. Skipping {step_name} step.")
        return load_func()
    else:
        print(f"{step_name.capitalize()} data not found. Running {step_name} step.")
        processed_data = process_func() if data is None else process_func(data)
        save_func(processed_data, file_path)
        return processed_data


def process_data(df, config_path):
    """
    Processes data by dynamically applying defined steps in a configuration file.

    Args:
        df (pd.DataFrame): The DataFrame to be processed.
        config_path (str): Path to the configuration file that defines steps for processing.

    Returns:
        pd.DataFrame: The DataFrame after processing.
    """
    config = load_config(config_path)
    steps_config = config.get("steps", [])
    processed_df = apply_steps(df, steps_config)
    return processed_df


def apply_steps(df, steps_config):
    """
    Applies a series of steps to a DataFrame based on a configuration.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        steps_config (list): A list of dictionaries, each containing the 'name' of the function to apply
                             and 'kwargs' for any arguments to pass to the function.

    Returns:
        pd.DataFrame: The processed DataFrame after all steps have been applied.
    """

    def apply_step(df, step):
        if not should_execute_step(step):
            print(f"Skipping step: {step['name']} as per the configuration.")
            return df
        module_name, function_name = step["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        kwargs = step.get("kwargs", {})
        return func(df, **kwargs)

    return reduce(apply_step, steps_config, df)
