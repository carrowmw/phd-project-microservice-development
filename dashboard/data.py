# dashboard/data.py

# Compare this snippet from apps/sensor_info.py:
import random
from datetime import datetime
import numpy as np
import pandas as pd

from config.paths import (
    get_daily_record_counts_path,
    get_completeness_metrics_path,
    get_freshness_metrics_path,
    get_evaluation_predictions_path,
    get_training_windows_path,
    add_root_to_path,
)

# pylint: disable=wrong-import-position
add_root_to_path()

from utils.data_helper import (
    save_data_to_file,
    load_data_from_file,
    find_tuple_by_first_element,
    create_file_path,
    pipeline_input_data_filename,
    pipeline_output_data_filename,
    load_sensor_list,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
    load_trained_models,
    load_evaluation_metrics,
)

from utils.config_helper import get_datetime_column, get_last_n_days


class CustomDashboardData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomDashboardData, cls).__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return
        print("CustomDashboardData.__init__() started...")
        try:
            self.data = load_raw_data()
            self.preprocessed_data = load_preprocessed_data()
            self.engineered_data = load_engineered_data()
            self.dataloaders = load_dataloaders()
            self.trained_models = load_trained_models()
            self.evaluation_metrics = load_evaluation_metrics()
        except FileNotFoundError:
            print("Data not found. Please run the pipeline to generate the data.")
        except AttributeError:
            print("Data not found. Please run the pipeline to generate the data.")
            return
        self.latest_data = [(tuple[0], tuple[1][-500:]) for tuple in self.data]
        self.sensors = load_sensor_list()
        self.active_sensors = [tuple[0] for tuple in self.data]
        self.datetime_column = get_datetime_column()
        self.last_n_days = get_last_n_days()
        self._initialised = True
        print("CustomDashboardData.__init__() completed.\n")

    def get_sensor_info(self) -> tuple:
        """
        Get the sensor information such as location from the sensor API for active sensors.

        Args:
            active_sensors (List): A list of active sensor names.

        Returns:
            Tuple: A tuple containing the latitude, longitude, and name of the active sensors.
        """

        active_sensor_info = self.sensors[
            self.sensors["Sensor Name"].isin(self.active_sensors)
        ]

        missing_sensors = set(self.active_sensors) - set(
            active_sensor_info["Sensor Name"]
        )
        if missing_sensors:
            print(
                f"Warning: The following active sensors are not found in the sensor dataframe: {', '.join(missing_sensors)}"
            )

        name = active_sensor_info["Sensor Name"]
        lat = active_sensor_info["Sensor Centroid Latitude"]
        lon = active_sensor_info["Sensor Centroid Longitude"]

        return lat, lon, name

    def read_or_compute_app_data(self, file_path, func):
        """
        Read or compute data for the dashboard.
        """
        # print("CustomDashboardData.read_or_compute_app_data()")
        app_data = load_data_from_file(file_path)
        if app_data is not None:
            return app_data

        app_data = func()
        save_data_to_file(file_path, app_data)
        return app_data

    def compute_daily_counts(self) -> list:
        """
        Get daily counts dataframes.

        Parameters:
        - list_of_dataframes (list): A list of tuples containing DataFrame name and DataFrame.

        Returns:
        list: A list of daily counts DataFrames.

        For each DataFrame in the list, this function calculates daily counts based on the 'Timestamp'
        column, creates a new DataFrame with 'Timestamp' and 'Count' columns, and appends it to a list.

        Example:
        get_daily_counts_dataframes([('Sensor1', df1), ('Sensor2', df2)])
        """
        daily_counts = []
        for df in self.data:
            sensor_name, df = df[0], df[1]
            # Group by the date part of the timestamp and count the rows
            df = (
                df.groupby(df[self.datetime_column].dt.date)
                .size()
                .reset_index(name="Count")
            )
            daily_counts.append((sensor_name, df))

        return daily_counts

    def get_completeness_graph_data(self) -> list:
        """
        Retrieves and formats application data for visualization in a missing data app.

        It fetches data based on the configurations specified in 'api_config.json', including
        the bounding box coordinates and the number of days for which data is required. If
        data for the given period exists locally, it is read from the storage; otherwise, it is
        requested from the sensor API and processed.

        Returns:
            list: A list of dataframes containing the daily counts of records for each sensor.
        """
        file_path = create_file_path(
            get_daily_record_counts_path(), pipeline_input_data_filename
        )
        print("CustomDashboardData.get_completeness_graph_data()")
        daily_counts = self.read_or_compute_app_data(
            file_path, self.compute_daily_counts
        )
        return daily_counts

    def compute_completeness_metrics(self):
        """
        Compute the completeness metrics for the data.

        Returns:
            str: A string containing the data completeness metric.
        """
        completeness_metrics = [
            (data[0], str(round(len(data[1]) / (self.last_n_days * 96) * 100, 2)) + "%")
            for data in self.data
        ]

        return completeness_metrics

    def get_completeness_metrics(self):
        """
        Create a metric showing the completeness of the data.

        Returns:
            str: A string containing the data completeness metric.
        """
        file_path = create_file_path(
            get_completeness_metrics_path(), pipeline_input_data_filename
        )
        print("CustomDashboardData.get_completeness_metrics()")
        completeness_metrics = self.read_or_compute_app_data(
            file_path, self.compute_completeness_metrics
        )

        # print("Completeness metrics: ", completeness_metrics)
        return completeness_metrics

    def compute_freshness_metrics(self):
        """
        Compute the freshness metrics for the data.

        Returns:
            str: A string containing the data freshness metric.
        """
        freshness_metrics = []
        for data in self.data:
            delta = datetime.now() - data[1][self.datetime_column].max()
            # rounded_delta = timedelta(seconds=round(delta.total_seconds()))
            # Format the timedelta string with days component
            days = delta.days
            hours, remainder = divmod(delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            delta_str = f"{days} days\n{hours:02d}:{minutes:02d}:{seconds:02d}"

            freshness_metrics.append((data[0], delta_str))

        return freshness_metrics

    def get_freshness_legend_tick_text(self):
        """
        Get the tick text for the freshness legend.

        Returns:
            Int: The maximum number of days since the data was last updated.
        """
        days = []
        for data in self.data:
            delta = datetime.now() - data[1][self.datetime_column].max()
            # rounded_delta = timedelta(seconds=round(delta.total_seconds()))
            # Format the timedelta string with days component
            day = delta.days
            days.append(day)
            max_day = max(days)

        return max_day

    def get_freshness_metrics(self):
        """
        Create a metric showing the freshness of the data.
        Returns:
            str: A string containing the data freshness metric.
        """
        file_path = create_file_path(
            get_freshness_metrics_path(), pipeline_input_data_filename
        )
        print("CustomDashboardData.get_freshness_metrics()")
        freshness_metrics = self.read_or_compute_app_data(
            file_path, self.compute_freshness_metrics
        )
        return freshness_metrics

    def compute_evaluation_predictions(self):
        """
        Compute the evaluation predictions for the sensors.

        Returns:
            List: A list of tuples containing the sensor name, the evaluation metric, and the prediction.
        """
        data = load_evaluation_metrics()
        evaluation_predictions = [(tuple[0], tuple[1], tuple[2]) for tuple in data]

        return evaluation_predictions

    def get_evaluation_predictions(self):
        """
        Get the evaluation data for the sensors.

        Returns:
            List: A list of tuples containing the sensor name, the evaluation metric, and the prediction.
        """
        file_path = create_file_path(
            get_evaluation_predictions_path(), pipeline_output_data_filename
        )
        print(file_path)
        print("CustomDashboardData.get_evaluation_predictions()")
        evaluation_predictions = self.read_or_compute_app_data(
            file_path, self.compute_evaluation_predictions
        )
        return evaluation_predictions

    def get_preprocessing_table(self, sensor_name=None):
        """
        Get the preprocessing data for the sensors.

        Returns:
            List: A list of tuples containing the sensor name, the preprocessing metric, and the prediction.
        """

        if sensor_name is None:
            sensor_name = self.active_sensors[0]
        print("CustomDashboardData.get_preprocessing_table()")
        data = load_preprocessed_data()
        data = find_tuple_by_first_element(data, sensor_name)
        assert data is not None, f"Expected data to be a pd.DataFrame, but got {data}"
        if data.empty is True:
            return pd.DataFrame()
        data = data.reset_index()
        return data

    def get_engineering_table(self, sensor_name=None):
        """
        Get the engineered data for the sensors.

        Returns:
            List: A list of tuples containing the sensor name, the engineering metric, and the prediction.
        """
        if sensor_name is None:
            sensor_name = self.active_sensors[0]
        print("CustomDashboardData.get_engineering_table()")
        data = load_engineered_data()
        data = find_tuple_by_first_element(data, sensor_name)
        assert data is not None, f"Expected data to be a pd.DataFrame, but got {data}"
        if data.empty is True:
            return pd.DataFrame()
        data = data.reset_index()
        return data

    def get_training_windows(self) -> list[list[np.ndarray]]:
        """
        Retrieves and formats application data for visualization in a missing data app.

        Returns:
            List[List[np.ndarray]]: A list of lists containing the training windows and
            labels for the LSTM.
        """
        print("CustomDashboardData.get_training_windows()")
        file_path = create_file_path(
            get_training_windows_path(), pipeline_output_data_filename
        )
        training_windows_data = load_data_from_file(file_path)
        if training_windows_data is not None:
            return training_windows_data

        training_windows_data = load_dataloaders()

        print("Unbatching...")
        app_data = [[], [], [], []]
        for _, (_, _, _, val_dataloader, _) in enumerate(training_windows_data):
            unbatched_input_feature = []
            unbatched_labels = []
            unbatched_eng_features = []
            dataloader = val_dataloader
            for batch in dataloader:
                features, labels = batch
                features_array, labels_array = features.numpy(), labels.numpy()
                labels_array = labels_array.reshape(-1, 1)

                _, _, no_of_features = features_array.shape

                # Iterate through each feature
                for i in range(no_of_features):
                    feature_data = features_array[:, :, i]

                    # Append the feature data to the unbatched_eng_features array
                    if len(unbatched_eng_features) <= i:
                        unbatched_eng_features.append(feature_data)
                    else:
                        unbatched_eng_features[i] = np.concatenate(
                            (unbatched_eng_features[i], feature_data), axis=0
                        )

                unbatched_input_feature.append(features_array[:, :, 0])
                unbatched_labels.append(labels_array)

            unbatched_input_feature = np.concatenate(unbatched_input_feature, axis=0)
            unbatched_labels = np.concatenate(unbatched_labels, axis=0)
            unbatched_eng_features = np.array(unbatched_eng_features)

            app_data[0].append(unbatched_input_feature)
            app_data[1].append(unbatched_labels)
            app_data[2].append(unbatched_eng_features)

        for pipeline_object in training_windows_data:
            sensor_name = pipeline_object[0]
            app_data[3].append(sensor_name)
        print("Unbatching complete")
        save_data_to_file(file_path, app_data)
        print(f"Traning windows data saved to {file_path}.")

        return app_data

    def get_random_sensor(self):
        """
        Get a random sensor from the list of active sensors.

        Returns:
            str: A random sensor name.
        """
        random.seed(42)
        sensor_name = random.choice(self.active_sensors)
        return sensor_name
