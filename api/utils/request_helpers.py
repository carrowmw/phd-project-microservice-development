# ./api/utils/request_helpers.py

"""
Utility functions related to API requests (e.g. handling errors, common functionality)
"""

import requests
from shapely import Polygon
from shapely.wkb import dumps


def handle_api_response(response):
    """
    Handles the API response and returns the JSON data if successful, otherwise raises
    a more specific exception.
    """
    try:
        if isinstance(response, requests.Response):
            response.raise_for_status()
            return response.json()
        else:
            raise ValueError("Invalid response type. Expected requests.Response.")
    except requests.exceptions.HTTPError as errh:
        raise ValueError(f"HTTP Error: {errh}") from errh
    except requests.exceptions.ConnectionError as errc:
        raise ValueError(f"Error Connecting: {errc}") from errc
    except requests.exceptions.Timeout as errt:
        raise ValueError(f"Timeout Error: {errt}") from errt
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Request Error: {err}") from err


def make_api_request(url, params=None, timeout=1000):
    """
    Makes a generic API request with error handling.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as errh:
        raise ValueError(f"HTTP Error: {errh}") from errh
    except requests.exceptions.Timeout as errt:
        raise ValueError(f"Timeout Error: {errt}") from errt
    except requests.exceptions.RequestException as err:
        raise ValueError(f"API request failed: {err}") from err


def create_polygon_wkb(coords: list):
    """
    Create a Well-Known Binary (WKB) representation of a polygon from given minimum and maximum longitude and latitude values.

    Args:
        min_lon (float): The minimum longitude of the polygon.
        min_lat (float): The minimum latitude of the polygon.
        max_lon (float): The maximum longitude of the polygon.
        max_lat (float): The maximum latitude of the polygon.

    Returns:
        str: A WKB representation of the polygon in hexadecimal format.
    """
    # create a shapely polygon using the provided coordinates
    min_lon, min_lat, max_lon, max_lat = coords[0], coords[1], coords[2], coords[3]
    polygon = Polygon(
        [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    )

    # convert polygon to WKB format
    polygon_wkb = dumps(polygon, hex=True)

    return polygon_wkb
