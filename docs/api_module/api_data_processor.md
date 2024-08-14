# Module: `api_data_processor.py`

The `api_data_processor.py` script handles the processing of data retrieved from the API. It defines the `APIDataProcessor` class, which is used to clean, preprocess, and manipulate the sensor data for further analysis. The APIDataProcessor requires a JSON config file with specified parameters.

Class: `APIDataProcessor`

The APIDataProcessor class initialises with configuration parameters and provides methods to execute various API requests and process the data.

## Initialisation

```python
class APIDataProcessor:
    def __init__(self):
        self.api_client = APIClient()
        self.config = get_api_config()
        self.window_size = get_window_size()
        self.min_df_length = get_min_df_length()
        self.ratio = get_min_df_length_to_window_size_ratio()
```

## Methods

* `execute_sensors_request() -> pd.DataFrame`
  * Executes the API request to fetch sensor data and converts it to a `DataFrame`.
  * Returns: `DataFrame` containing sensor data.
* `execute_sensor_types_request() -> pd.DataFrame`
  * Executes the API request to fetch sensor types and converts it to a `DataFrame`.
  * Returns: `DataFrame` containing sensor types.
* `execute_themes_request() -> pd.DataFrame`
  * Executes the API request to fetch themes and converts it to a `DataFrame`.
  * Returns: `DataFrame` containing themes.
* `execute_variables_request() -> pd.DataFrame`
  * Executes the API request to fetch variables and converts it to a `DataFrame`.
  * Returns: `DataFrame` containing variables.
* `print_api_response_information(sensor_name: str, index: int, total_sensors: int)`
  * Prints information about the API response for a specific sensor.
* `process_each_sensor(sensor_name: str, index: int, total_sensors: int) -> tuple`
  * Processes the data for a specific sensor.
  * Parameters:
    * `sensor_name`: The name of the sensor.
    * `index`: The index of the sensor in the list.
    * `total_sensors`: The total number of sensors.
  * Returns: Tuple containing sensor name and `DataFrame` with sensor data.
* `print_sensor_data_metrics(list_of_dataframes, series_of_sensor_names)`
  * Prints metrics about the processed sensor data.
* `process_sensor_data_parallel(sensor_names: pd.Series) -> list`
  * Processes the data for sensors in parallel.
  * Parameters:
    * `sensor_names`: Series containing the names of the sensors.
  * Returns: List of tuples containing sensor names and `DataFrames` with sensor data.
* `execute_data_request() -> list`
  * Executes the data request and processes the data.
  * Returns: List of tuples containing sensor names and `DataFrames` with sensor data.

## Example Usage

```python
from api_data_processor import APIDataProcessor

data_processor = APIDataProcessor()

# Execute sensor request and get the DataFrame
sensors_df = data_processor.execute_sensors_request()

# Execute sensor types request and get the DataFrame
sensor_types_df = data_processor.execute_sensor_types_request()

# Execute themes request and get the DataFrame
themes_df = data_processor.execute_themes_request()

# Execute variables request and get the DataFrame
variables_df = data_processor.execute_variables_request()

# Process sensor data
sensor_data = data_processor.execute_data_request()
```