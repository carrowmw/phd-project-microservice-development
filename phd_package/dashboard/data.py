# data.py

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from phd_package.pipeline import Pipeline
from .utils.transformation_helper import unbatch_dataloaders_to_numpy
from .utils.error_helper import handle_data_errors
from ..config.paths import *
from ..utils.data_helper import (
    load_sensor_list,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloader,
    load_trained_models,
    load_test_metrics,
)

from ..utils.config_helper import (
    get_datetime_column,
    get_n_days,
    get_anomaly_std,
)


class DataLoader:
    @handle_data_errors(default_return=lambda: None)
    def load_data(self):
        return {
            "raw_data": load_raw_data(),
            "preprocessed_data": load_preprocessed_data(),
            "engineered_data": load_engineered_data(),
            "dataloader": load_dataloader(),
            "trained_models": load_trained_models(),
            "test_metrics": load_test_metrics(),
        }


class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.datetime_column = get_datetime_column()
        self.n_days = get_n_days()
        self.anomaly_std = get_anomaly_std()

    def process_data(self):
        return {
            "latest_data": self._process_latest_data(),
            "sensors": self._process_sensors(),
            "active_sensors": self._process_active_sensors(),
            "trainable_sensors": self._process_trainable_sensors(),
        }

    def _process_latest_data(self):
        return {sensor: data.tail(500) for sensor, data in self.data["raw_data"]}

    def _process_sensors(self):
        return load_sensor_list()

    def _process_active_sensors(self):
        return [sensor for sensor, _ in self.data["raw_data"]]

    def _process_trainable_sensors(self):
        return [sensor for sensor, _, _, _, _ in self.data["dataloader"]]


class MetricsCalculator:
    def __init__(self, data, processed_data):
        self.data = data
        self.processed_data = processed_data
        self.n_days = get_n_days()
        self.anomaly_std = get_anomaly_std()

    def calculate_metrics(self):
        try:
            return {
                "completeness": self._calculate_completeness_metrics(),
                "freshness": self._calculate_freshness_metrics(),
                "daily_counts": self._calculate_daily_counts(),
                "test_predictions": self._calculate_test_predictions(),
                "train_metrics": self._calculate_train_metrics(),
                "anomalies": self._calculate_anomalies(),
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                "completeness": {},
                "freshness": {},
                "daily_counts": {},
                "test_predictions": {},
                "train_metrics": {},
                "anomalies": {},
            }

    def _calculate_completeness_metrics(self):
        metrics = {}
        try:
            max_length = max(
                (len(data) for _, data in self.data["raw_data"]), default=1
            )
            for sensor, data in self.data["raw_data"]:
                try:
                    length = len(data)
                    expected_records = self.n_days * 96  # Expected number of records
                    if expected_records == 0:
                        raise ValueError("Expected records cannot be zero")

                    float_value = length / expected_records
                    # Avoid division by zero for normalization
                    norm_value = float_value / max_length if max_length > 0 else 0

                    metrics[sensor] = {
                        "float": float_value,
                        "norm": norm_value,
                        "string": f"{round(float_value * 100, 2)}%",
                    }
                except Exception as e:
                    print(
                        f"Error calculating completeness for sensor {sensor}: {str(e)}"
                    )
                    metrics[sensor] = {"float": 0, "norm": 0, "string": "0%"}
            return metrics
        except Exception as e:
            print(f"Error in completeness metrics calculation: {str(e)}")
            return {}

    def _calculate_freshness_metrics(self):
        metrics = {}
        try:
            for sensor, data in self.data["raw_data"]:
                try:
                    most_recent = data[get_datetime_column()].max()
                    time_diff = (datetime.now() - most_recent).total_seconds()
                    total_seconds = timedelta(days=self.n_days).total_seconds()

                    # Avoid division by zero
                    if total_seconds == 0:
                        raise ValueError("Total seconds cannot be zero")

                    float_value = time_diff / total_seconds

                    # Handle case where log of zero or negative number
                    if float_value <= 0:
                        log_value = float("-inf")
                    else:
                        log_value = np.log(float_value)

                    metrics[sensor] = {
                        "float": float_value,
                        "log": log_value,
                        "string": self._format_time_difference(most_recent),
                    }
                except Exception as e:
                    print(f"Error calculating freshness for sensor {sensor}: {str(e)}")
                    metrics[sensor] = {
                        "float": 0,
                        "log": float("-inf"),
                        "string": "No data",
                    }

            # Calculate normalized values only if we have valid metrics
            if metrics:
                max_float = max(m["float"] for m in metrics.values())
                log_values = [
                    m["log"] for m in metrics.values() if m["log"] != float("-inf")
                ]
                if log_values:
                    min_log = min(log_values)
                    max_log = max(log_values)
                    log_range = max_log - min_log if max_log != min_log else 1

                    for sensor in metrics:
                        metrics[sensor]["norm"] = (
                            metrics[sensor]["float"] / max_float
                            if max_float != 0
                            else 0
                        )
                        metrics[sensor]["norm_log"] = (
                            (metrics[sensor]["log"] - min_log) / log_range
                            if metrics[sensor]["log"] != float("-inf")
                            else 0
                        )

            return metrics
        except Exception as e:
            print(f"Error in freshness metrics calculation: {str(e)}")
            return {}

    def _format_time_difference(self, most_recent):
        """Helper method to format time difference string"""
        try:
            time_diff = datetime.now() - most_recent
            days = time_diff.days
            hours = time_diff.seconds // 3600
            minutes = (time_diff.seconds % 3600) // 60
            seconds = time_diff.seconds % 60
            return f"{days} days\n{hours}:{minutes:02d}:{seconds:02d}"
        except Exception as e:
            print(f"Error formatting time difference: {str(e)}")
            return "Error calculating time"

    def _calculate_daily_counts(self):
        try:
            return {
                sensor: data.groupby(data[get_datetime_column()].dt.date)
                .size()
                .reset_index(name="Count")
                for sensor, data in self.data["raw_data"]
                if not data.empty
            }
        except Exception as e:
            print(f"Error calculating daily counts: {str(e)}")
            return {}

    def _calculate_test_predictions(self):
        try:
            return {
                sensor: (predictions, labels)
                for sensor, predictions, labels, _ in self.data["test_metrics"]
                if predictions is not None and labels is not None
            }
        except Exception as e:
            print(f"Error calculating test predictions: {str(e)}")
            return {}

    def _calculate_train_metrics(self):
        try:
            return {
                sensor: (train_loss, val_metrics)
                for sensor, _, _, train_loss, val_metrics in self.data["trained_models"]
                if train_loss is not None and val_metrics is not None
            }
        except Exception as e:
            print(f"Error calculating train metrics: {str(e)}")
            return {}

    def _calculate_anomalies(self):
        anomalies = {}
        try:
            if self.anomaly_std is None:
                raise ValueError("anomaly_std is None")

            for sensor, predictions, labels, _ in self.data["test_metrics"]:
                try:
                    if predictions is None or labels is None:
                        continue

                    df = pd.DataFrame(
                        {
                            "predictions": predictions.flatten(),
                            "labels": labels.flatten(),
                        }
                    )

                    df["errors"] = df["predictions"] - df["labels"]

                    # Check for valid statistics before calculating threshold
                    error_mean = df["errors"].mean()
                    error_std = df["errors"].std()

                    if pd.isna(error_mean) or pd.isna(error_std):
                        raise ValueError(f"Invalid statistics for sensor {sensor}")

                    threshold = error_mean + self.anomaly_std * error_std
                    df["anomaly"] = df["errors"] > threshold
                    anomalies[sensor] = df

                except Exception as e:
                    print(f"Error calculating anomalies for sensor {sensor}: {str(e)}")
                    # Create empty DataFrame with expected columns
                    anomalies[sensor] = pd.DataFrame(
                        columns=["predictions", "labels", "errors", "anomaly"]
                    )

            return anomalies

        except Exception as e:
            print(f"Error in anomaly calculation: {str(e)}")
            return {}


class CustomDashboardData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomDashboardData, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print("CustomDashboardData.__init__() started...")
        self._load_and_process_data()
        self._initialized = True
        print("CustomDashboardData.__init__() completed.\n")

    def _load_and_process_data(self):
        data_loader = DataLoader()
        self.data = data_loader.load_data()

        data_processor = DataProcessor(self.data)
        self.processed_data = data_processor.process_data()

        metrics_calculator = MetricsCalculator(self.data, self.processed_data)
        self.sensor_metrics = metrics_calculator.calculate_metrics()

    def get_sensor_info(self):
        active_sensor_info = self.processed_data["sensors"][
            self.processed_data["sensors"]["Sensor Name"].isin(
                self.processed_data["active_sensors"]
            )
        ]
        return (
            active_sensor_info["Sensor Centroid Latitude"],
            active_sensor_info["Sensor Centroid Longitude"],
            active_sensor_info["Sensor Name"],
        )

    def get_sensor_metrics(self, metric_type):
        return self.sensor_metrics.get(metric_type, {})

    def get_preprocessing_table(self, sensor_name=None):
        sensor_name = sensor_name or self.processed_data["active_sensors"][0]
        data = next(
            (data for s, data in self.data["preprocessed_data"] if s == sensor_name),
            None,
        )
        return pd.DataFrame() if data is None or data.empty else data.reset_index()

    def get_engineering_table(self, sensor_name=None):
        sensor_name = sensor_name or self.processed_data["active_sensors"][0]
        data = next(
            (data for s, data in self.data["engineered_data"] if s == sensor_name), None
        )
        return pd.DataFrame() if data is None or data.empty else data.reset_index()

    def get_training_windows(self):
        app_data = [[], [], [], []]
        for sensor_name, _, _, val_dataloader, _ in self.data["dataloader"]:
            unbatched_input_feature, unbatched_labels, unbatched_eng_features = (
                unbatch_dataloaders_to_numpy(val_dataloader)
            )
            app_data[0].append(unbatched_input_feature)
            app_data[1].append(unbatched_labels)
            app_data[2].append(unbatched_eng_features)
            app_data[3].append(sensor_name)
        return app_data

    def get_metadata(self):
        return {
            "Number of Sensors": len(self.processed_data["sensors"]),
            "Number of Active Sensors": len(self.processed_data["active_sensors"]),
            "Number of Trainable Sensors": len(
                self.processed_data["trainable_sensors"]
            ),
            "Length of Sensor Info:": len(self.get_sensor_info()[0]),
            "Length of Completeness Metrics": len(self.sensor_metrics["completeness"]),
            "Length of Freshness Metrics": len(self.sensor_metrics["freshness"]),
            "Length of Daily Counts": len(self.sensor_metrics["daily_counts"]),
            "Length of Test Metrics": len(self.data["test_metrics"]),
        }

    def get_random_sensor(self):
        import random

        random.seed(42)
        return random.choice(self.processed_data["trainable_sensors"])
