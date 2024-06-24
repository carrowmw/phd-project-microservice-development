# Create a map showing sensor locations to use in the dashboard

# Path: apps/sensor_dashboard.py
# Compare this snippet from apps/sensor_info.py:
import random
from datetime import datetime, timedelta
import numpy as np
from phd_project.src.utils.app_utils import (
    save_data_to_file,
    load_data_from_file,
)

from phd_project.src.data_processing.execute_requests import get_daily_counts_dataframes

from phd_project.src.pipeline import download_sensor_list, read_or_download_raw_data
from phd_project.src.utils.app_utils import find_tuple_by_first_element
from phd_project.src.utils.general_utils import get_last_n_days_from_config
from phd_project.src.utils.fstore_utils import (
    create_file_path_from_config,
    load_evaluation_metrics,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
)

from phd_project.src.utils.general_utils import get_datetime_column_from_config


def get_list_of_active_sensors():
    """
    Get the list of active sensors based on the returned raw sensor data.

    Returns:
        List: A list of active sensors.
    """
    raw_data = read_or_download_raw_data()
    active_sensors = [tuple[0] for tuple in raw_data]
    # print(f"Number of active sensors: {len(active_sensors)}")
    return active_sensors


# Get key information from sensor api
def get_sensor_info():
    """
    Get the sensor information such as location from the sensor API for active sensors.

    Args:
        active_sensors (List): A list of active sensor names.

    Returns:
        Tuple: A tuple containing the latitude, longitude, and name of the active sensors.
    """
    sensors = download_sensor_list()
    active_sensors = get_list_of_active_sensors()
    active_sensor_info = sensors[sensors["Sensor Name"].isin(active_sensors)]

    missing_sensors = set(active_sensors) - set(active_sensor_info["Sensor Name"])
    if missing_sensors:
        print(
            f"Warning: The following active sensors are not found in the sensor dataframe: {', '.join(missing_sensors)}"
        )

    name = active_sensor_info["Sensor Name"]
    lat = active_sensor_info["Sensor Centroid Latitude"]
    lon = active_sensor_info["Sensor Centroid Longitude"]

    return lat, lon, name


def get_latest_sensor_data():
    """
    Get the latest sensor data from the sensor API.

    Returns:
        DataFrame: A dataframe containing the latest sensor data.
    """
    raw_data = read_or_download_raw_data()
    latest_data = [(tuple[0], tuple[1][-500:]) for tuple in raw_data]
    # print(f"Latest data {latest_data}")
    return latest_data


def get_completeness_graph_data() -> list:
    """
    Retrieves and formats application data for visualization in a missing data app.

    It fetches data based on the configurations specified in 'api_config.json', including
    the bounding box coordinates and the number of days for which data is required. If
    data for the given period exists locally, it is read from the storage; otherwise, it is
    requested from the sensor API and processed.

    Returns:
        list: A list of dataframes containing the daily counts of records for each sensor.
    """
    file_path = create_file_path_from_config("app_data/daily_record_counts")
    app_data = load_data_from_file(file_path)
    if app_data is not None:
        return app_data

    data_list = load_preprocessed_data()
    app_data = get_daily_counts_dataframes(data_list)
    save_data_to_file(file_path, app_data)
    # print(f"Missing data {app_data}")
    return app_data


def get_completeness_metrics():
    """
    Create a metric showing the completeness of the data.

    Returns:
        str: A string containing the data completeness metric.
    """
    file_path = create_file_path_from_config("app_data/completeness_metrics")
    completeness_metrics = load_data_from_file(file_path)
    if completeness_metrics is not None:
        print("Completeness metrics loaded from file.")
        completeness_metrics = [("sensor_id", completeness_metrics)]
        return completeness_metrics

    data_list = read_or_download_raw_data()

    last_n_days = get_last_n_days_from_config()

    completeness_metrics = [
        (data[0], str(round(len(data[1]) / (last_n_days * 96) * 100, 2)) + "%")
        for data in data_list
    ]

    return completeness_metrics


def get_freshness_metrics():
    """
    Create a metric showing the freshness of the data.
    Returns:
        str: A string containing the data freshness metric.
    """
    file_path = create_file_path_from_config("app_data/freshness_metrics")
    freshness_metrics = load_data_from_file(file_path)
    if freshness_metrics is not None:
        print("Freshness metrics loaded from file.")
        return freshness_metrics

    data_list = read_or_download_raw_data()
    datetime_column = get_datetime_column_from_config()

    freshness_metrics = []
    for data in data_list:
        delta = datetime.now() - data[1][datetime_column].max()
        # rounded_delta = timedelta(seconds=round(delta.total_seconds()))
        # Format the timedelta string with days component
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        delta_str = f"{days} days\n{hours:02d}:{minutes:02d}:{seconds:02d}"

        freshness_metrics.append((data[0], delta_str))

    return freshness_metrics


def get_random_sensor():
    """
    Get a random sensor from the list of active sensors.

    Returns:
        str: A random sensor name.
    """
    random.seed(42)
    active_sensors = get_list_of_active_sensors()
    sensor_name = random.choice(active_sensors)
    return sensor_name


def get_evaluation_predictions_data():
    """
    Get the evaluation data for the sensors.

    Returns:
        List: A list of tuples containing the sensor name, the evaluation metric, and the prediction.
    """
    file_path = create_file_path_from_config("app_data/evaluation_predictions")
    app_data = load_data_from_file(file_path)
    if app_data is not None:
        return app_data
    evaluation_metrics_list = load_evaluation_metrics()
    evaluation_data = [
        (tuple[0], tuple[1], tuple[2]) for tuple in evaluation_metrics_list
    ]
    save_data_to_file(file_path, evaluation_data)

    return evaluation_data


def get_preprocessing_table(sensor_name=None):
    """
    Get the preprocessing data for the sensors.

    Returns:
        List: A list of tuples containing the sensor name, the preprocessing metric, and the prediction.
    """

    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]
    data_list = load_preprocessed_data()
    data = find_tuple_by_first_element(data_list, sensor_name)
    assert data is not None, f"Expected data to be a pd.DataFrame, but got {data}"
    if data.empty is True:
        return "No data"
    data = data.reset_index()
    return data


def get_engineering_table(sensor_name=None):
    """
    Get the engineered data for the sensors.

    Returns:
        List: A list of tuples containing the sensor name, the engineering metric, and the prediction.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]
    data_list = load_engineered_data()
    data = find_tuple_by_first_element(data_list, sensor_name)
    assert data is not None, f"Expected data to be a pd.DataFrame, but got {data}"
    if data.empty is True:
        return "No data"
    data = data.reset_index()
    return data


if __name__ == "__main__":
    get_evaluation_predictions_data()


def get_training_windows_data() -> list[list[np.ndarray]]:
    """
    Retrieves and formats application data for visualization in a missing data app.

    Returns:
        List[List[np.ndarray]]: A list of lists containing the training windows and
        labels for the LSTM.
    """
    file_path = create_file_path_from_config("app_data/training_windows")
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
    print("Traning windows data saved to file.")

    return app_data
