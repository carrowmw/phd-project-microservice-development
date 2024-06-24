import os
import json
from datetime import date
import pickle

from phd_project.src.utils.polygon_utils import create_wkb_polygon
from phd_project.src.pipeline import download_sensor_list


def save_data_to_file(file_path, data):
    """
    Saves the given data to a file at the specified path, creating any necessary directories.

    Args:
        file_path (str): The path where the data should be saved.
        data (Any): The data to be saved.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(file_path):
    """
    Loads data from the specified file path if it exists.

    Args:
        file_path (str): The path of the file to load data from.

    Returns:
        The data loaded from the file if it exists, otherwise None.
    """
    if os.path.exists(file_path):
        print("\nReading in app data from local storage\n")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


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
