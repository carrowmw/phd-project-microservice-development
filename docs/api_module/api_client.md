# Module: `api_client.py`

The `api_client.py` script provides the core functionalities for making API requests and handling responses. It defines the APIClient class, which is used to interact with various endpoints of the API.

Class: `APIClient`

The APIClient class initialises with the base URL and token required for API access. It has several methods to interact with different endpoints defined in the api.json configuration file.

## Initialisation

```python
class APIClient:
    def __init__(self, base_url: str, token: str, endpoints: dict):
        self.base_url = base_url
        self.token = token
        self.endpoints = endpoints
```
## Methods

* `get_sensors(theme: str, polygon_wkb: str) -> dict`
  * Fetches sensors based on the theme and polygon WKB (Well-Known Binary) format.
  * **Parameters**:
    * `theme`: The theme of the data (run `APIClient.get_themes()` for a full list).
    * `polygon_wkb`: The polygon WKB defining the area of interest.
  * Returns: JSON response containing sensor data.
* `get_sensor_types() -> dict`
  * Fetches available sensor types.
  * Returns: JSON response containing sensor types.
* `get_themes()` -> dict
  * Fetches available themes.
  * Returns: JSON response containing themes.
* `get_variables(theme: str) -> dict`
  * Fetches variables based on the theme.
  * Parameters:
    * `theme`: The theme of the data (e.g., “People”).
  * Returns: JSON response containing variables.
* `get_data(sensor_name: str, starttime: str, endtime: str) -> dict`
  * Fetches data for a specific sensor within the given time range.
  * Parameters:
    * `sensor_name`: The name of the sensor.
    * `starttime`: Start time for the data in `YYYYMMDD` format.
    * `endtime`: End time for the data in `YYYYMMDD` format.
  * Returns: JSON response containing sensor data.

## Example Usage

```python
from api_client import APIClient

api_config = {
    "base_url": "https://newcastle.urbanobservatory.ac.uk/api/v1.1/",
    "token": "your_api_token",
    "endpoints": {
        # endpoint details
    }
}

client = APIClient(api_config["base_url"], api_config["token"], api_config["endpoints"])

# Fetch sensors
sensors = client.get_sensors("People", "POLYGON((...))")

# Fetch sensor types
sensor_types = client.get_sensor_types()

# Fetch themes
themes = client.get_themes()

# Fetch variables
variables = client.get_variables("People")

# Fetch data
data = client.get_data("sensor_name", "20220724", "20240724")
```