# api/api_data_processor.py
import concurrent.futures
import pandas as pd

from config.paths import add_root_to_path

add_root_to_path()

# pylint: disable=<C0413:wrong-import-position>
from utils.config_helper import (
    get_datetime_column,
    get_api_config,
    get_window_size,
    get_min_df_length,
    get_min_df_length_to_window_size_ratio,
)
from utils.data_transformation_helper import json_to_dataframe

from .api_client import APIClient


class APIDataProcessor:
    """
    Class to process data from the Urban Observatory API.
    """

    def __init__(self):
        self.api_client = APIClient()
        self.config = get_api_config()
        self.window_size = get_window_size()
        self.min_df_length = get_min_df_length()
        self.ratio = get_min_df_length_to_window_size_ratio()

    def execute_sensors_request(self) -> pd.DataFrame:
        data = self.api_client.get_sensors()
        df = json_to_dataframe(data["sensors"])
        return df

    def execute_sensor_types_request(self) -> pd.DataFrame:
        data = self.api_client.get_sensor_types()
        df = json_to_dataframe(data["Variables"])
        return df

    def execute_themes_request(self) -> pd.DataFrame:
        data = self.api_client.get_themes()
        df = json_to_dataframe(data["Themes"])
        return df

    def execute_variables_request(self) -> pd.DataFrame:
        data = self.api_client.get_variables()
        df = json_to_dataframe(data["Variables"])
        return df

    def print_api_response_information(
        self, sensor_name: str, index: int, total_sensors: int
    ):
        """
        Prints information about the API response.
        """
        print(
            f"Processing sensor {index + 1} of {total_sensors}: {sensor_name}",
            end="\n",
            flush=True,
        )
        if index + 1 == total_sensors:
            print("Finished processing sensors.")

    def process_each_sensor(
        self, sensor_name: str, index: int, total_sensors: int
    ) -> tuple:
        """
        Processes the data for a specific sensor.
        """
        response = self.api_client.get_data(sensor_name)
        print(f"Response: {response}")
        if response and len(response["sensors"]) > 0:
            data = response["sensors"][0]["data"]
            # print(f"Data: {data}")
            if data and len(data) > self.min_df_length:
                self.print_api_response_information(sensor_name, index, total_sensors)
                # print(f"Length of data: {len(data)}")
                df = json_to_dataframe(data)
                datetime_column = get_datetime_column, get_api_config()
                df[datetime_column] = pd.to_datetime(df[datetime_column])

                return sensor_name, df

            self.print_api_response_information(sensor_name, index, total_sensors)
            print(
                f"Sensor found but less than specified min_df_length: {self.min_df_length}.",
                end="\n\n",
                flush=False,
            )
            return None

        self.print_api_response_information(sensor_name, index, total_sensors)
        print(
            "Sensor not found.",
            end="\n\n",
            flush=False,
        )
        return None

    def print_sensor_data_metrics(self, list_of_dataframes, series_of_sensor_names):
        active_sensor_count = len(list_of_dataframes)
        empty_sensor_count = len(series_of_sensor_names) - active_sensor_count
        empty_sensor_perc = empty_sensor_count / len(series_of_sensor_names)

        print(
            f"\n Percentage Empty Sensors:   \n     {100*round(empty_sensor_perc, 2)}%"
        )
        print(f"\n Count of Empty Sensors:     \n     {empty_sensor_count}")
        print(f"\n Count of Active Sensors:    \n     {active_sensor_count}")

    def process_sensor_data_parallel(self, sensor_names: pd.Series) -> list:
        """
        Processes the data for a specific sensor in parallel.
        """
        list_of_dataframes = []
        total_sensors = len(sensor_names)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda x: self.process_each_sensor(x[1], x[0], total_sensors),
                    sensor_names.items(),
                )
            )
            for result in results:
                if result:
                    list_of_dataframes.append(result)

        self.print_sensor_data_metrics(list_of_dataframes, sensor_names)

        return list_of_dataframes

    def execute_data_request(self) -> list:
        assert (
            self.window_size * self.ratio <= self.min_df_length
        ), f"window_size must be less than {self.min_df_length/self.ratio} for the specified min_length_df value."
        sensors = self.execute_sensors_request()
        sensor_names = sensors["Sensor Name"]
        dfs = self.process_sensor_data_parallel(sensor_names)

        return dfs
