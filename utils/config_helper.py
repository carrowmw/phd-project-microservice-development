import json
from shapely import Polygon
from shapely.wkb import dumps
from config.paths import (
    get_pipeline_config_path,
    get_api_config_path,
    get_query_config_path,
)


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


def get_kwargs(config, kwarg):
    """
    Retrieves the keyword arguments for a specific key from the configuration dictionary.

    Args:
        config (dict): The full configuration dictionary.
        key (str): The key to retrieve the keyword arguments for.

    Returns:
        dict: The keyword arguments for the specified key. Returns an empty dictionary if the key is not found.
    """
    return config.get(kwarg, {})


def get_window_size():
    """
    Retrieves the window size from the pipeline configuration.

    Returns:
        int: The window size.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "window_size")


def get_last_n_days():
    """
    Retrieves the number of days from the query configuration.

    Returns:
        int: The number of days.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return get_kwargs(query_config, "last_n_days")


def get_coords():
    """
    Retrieves the coordinates for the query from the query configuration.

    Returns:
        tuple: The coordinates for the query.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return get_kwargs(query_config, "coords")


def get_wkb_polygon():
    coords = get_coords()
    min_lon = coords[0]
    min_lat = coords[1]
    max_lon = coords[2]
    max_lat = coords[3]
    # create a shapely polygon using the provided coordinates
    polygon = Polygon(
        [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    )

    # convert polygon to WKB format
    wkb_polygon = dumps(polygon, hex=True)

    return wkb_polygon
