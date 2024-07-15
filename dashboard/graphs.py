# dashboard/graphs.py

"""
This module contains functions for creating the graphs and metrics that are displayed on the dashboard.

It calls functions from the `dashboard_data` module to fetch the data required for the graphs and metrics.

It contains the following functions:
- `create_completeness_metric`: Creates a metric showing the completeness of the data.
- `create_map_fig`: Creates a map showing sensor locations.
- `create_missing_data_graph`: Creates a graph showing missing data.
- `create_latest_data_graph`: Creates a graph showing the latest data.
- `create_evaluation_graph`: Creates a graph showing evaluation predictions.
- `create_training_windows_graph`: Creates a graph showing training windows.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.paths import get_mapbox_access_token_path

from utils.config_helper import (
    get_last_n_days,
    get_horizon,
)
from utils.data_helper import find_tuple_by_first_element
from dashboard.data import CustomDashboardData
from dashboard.utils.color_helper import (
    base_colors,
    accent_colors,
    completeness_color_scale,
    freshness_color_scale,
    blended_color_scale,
    map_data_to_colors,
)


class PipelineChartCreator:
    """
    Class to create the graphs and metrics for the dashboard.

    Attributes:
        dashboard_data (CustomDashboardData): An instance of the CustomDashboardData class.
    """

    def __init__(self):
        self.dashboard_data = CustomDashboardData()
        self.completeness_metrics = self.dashboard_data.get_completeness_metrics()
        self.freshness_metrics = self.dashboard_data.get_freshness_metrics()
        self.latest_sensor_data = self.dashboard_data.latest_data
        self.evaluation_predictions = self.dashboard_data.get_evaluation_predictions()
        self.training_windows = self.dashboard_data.get_training_windows()
        self.sensor_info = self.dashboard_data.get_sensor_info()
        self.completeness_graph_data = self.dashboard_data.get_completeness_graph_data()
        self.first_sensor_in_list = self.dashboard_data.active_sensors[0]
        self.last_n_days = get_last_n_days()

    def no_data_fig(self, sensor_name):
        """
        Create a figure with a message indicating no data available.

        Args:
            sensor_name (str): The name of the sensor for which no data is available.

        Returns:
            Figure: A plotly figure with a message indicating no data available.
        """
        # Return a figure with a message indicating no data available
        no_data_fig = go.Figure()
        no_data_fig.update_layout(
            title=f"No data available for sensor: {sensor_name}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20),
                )
            ],
        )

        return no_data_fig

    def create_completeness_metric(self, sensor_name=None):
        """
        Create a metric showing the completeness of the data.

        Args:
            sensor_name (str): The name of the sensor to create the metric for.

        Returns:
            str: A string containing the data completeness metric.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data_list = self.completeness_metrics
        completeness_value = find_tuple_by_first_element(data_list, sensor_name)
        if completeness_value is None:
            return "No data"
        return completeness_value

    def create_freshness_metric(self, sensor_name=None):
        """
        Create a metric showing the freshness of the data.

        Args:
            sensor_name (str): The name of the sensor to create the metric for.

        Returns:
            str: A string containing the data freshness metric.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data_list = self.freshness_metrics
        freshness_value = find_tuple_by_first_element(data_list, sensor_name)
        if freshness_value is None:
            return "No data"
        return freshness_value

    def create_records_per_day_graph(self, sensor_name=None) -> go.Figure:
        """
        Create a graph showing missing data.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the missing data graph.
        """

        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data = find_tuple_by_first_element(self.completeness_graph_data, sensor_name)
        if data is None:
            return self.no_data_fig(sensor_name)

        today = datetime.now()
        x_max = today.strftime("%Y-%m-%d")
        x_min = (today - timedelta(days=self.last_n_days)).strftime("%Y-%m-%d")

        assert isinstance(
            data, pd.DataFrame
        ), f"Expected data to be a pd.DataFrame, but got {type(data)}"

        missing_data_fig = go.Figure()
        missing_data_fig.add_trace(
            go.Bar(
                x=data["Timestamp"],
                y=data["Count"],
                name=sensor_name,
                marker=dict(color=base_colors["Mid Blue"]),
            )
        )

        missing_data_fig.update_layout(
            title={
                "text": "Actual records per day",
                "font": {"size": 20},
                "x": 0.5,  # Center the title horizontally
            },
            xaxis=dict(
                title="Date",
                range=[x_min, x_max],  # Set the x-axis range
            ),
            yaxis=dict(title="Total Records", range=[0, 96]),
        )

        return missing_data_fig

    def create_latest_data_graph(self, sensor_name=None) -> go.Figure:
        """
        Create a graph showing the latest data.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the latest data graph.
        """

        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data = find_tuple_by_first_element(self.latest_sensor_data, sensor_name)
        if data is None:
            return self.no_data_fig(sensor_name)

        assert isinstance(
            data, pd.DataFrame
        ), f"Expected data to be a pd.DataFrame, but got {type(data)}"

        latest_data_fig = go.Figure()
        latest_data_fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["Value"],
                name=sensor_name,
                marker=dict(color=base_colors["Mid Blue"]),
            )
        )

        latest_data_fig.update_layout(
            title={
                "text": "Last 500 records",
                "font": {"size": 20},
                "x": 0.5,  # Center the title horizontally
            },
            xaxis=dict(
                title="Date",
            ),
            yaxis=dict(
                title="Value",
            ),
        )

        return latest_data_fig

    def create_evaluation_graph(self, sensor_name=None) -> go.Figure:
        """
        Create a graph showing evaluation predictions.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the evaluation predictions graph.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data = find_tuple_by_first_element(
            self.evaluation_predictions, sensor_name, n_tuples=2
        )
        if data is None:
            return self.no_data_fig(sensor_name)
        assert isinstance(
            data, tuple
        ), f"Expected data to be a tuple, but got {type(data)}"

        predictions = [item[0] for item in data[0]]
        labels = [item[0] for item in data[1]]

        evaluation_fig = go.Figure()
        evaluation_fig.add_trace(
            go.Scatter(
                x=list(range(len(labels))),
                y=predictions,
                mode="lines+markers",
                name="Predictions",
            )
        )
        evaluation_fig.add_trace(
            go.Scatter(
                x=list(range(len(labels))),
                y=labels,
                mode="lines+markers",
                name="Labels",
            )
        )

        evaluation_fig.update_layout(
            title={
                "text": "Evaluation Predictions",
                "font": {"size": 20},
                "x": 0.5,  # Center the title horizontally
            },
            xaxis=dict(
                title="Labels",
            ),
            yaxis=dict(
                title="Predictions",
            ),
        )

        return evaluation_fig

    def create_training_windows_graph(self, sensor_name=None):
        """
        Create a graph showing training windows.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the training windows graph.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        horizon = get_horizon()
        data = self.training_windows
        if data is None or sensor_name not in data[3]:
            # Return a figure with a message indicating no data available
            return self.no_data_fig(sensor_name)
        assert isinstance(
            data, list
        ), f"Expected data to be a list, but got {type(data)}"

        sensor_index = data[3].index(sensor_name)

        features = data[0][sensor_index]
        labels = data[1][sensor_index]
        eng_features_list = data[2][sensor_index]

        current_index = 0
        x_values = list(range(len(features[current_index])))
        y_values = list(features[current_index])
        label_value = labels[current_index]

        figure = {
            "data": [
                go.Scatter(x=x_values, y=y_values, mode="lines", name="Input Feature"),
                go.Scatter(
                    x=[len(x_values) + horizon],
                    y=label_value,
                    mode="markers",
                    name="Label",
                ),
            ],
            "layout": {
                "xaxis": {"title": "X-axis"},
                "yaxis": {"title": "Y-axis"},
            },
        }

        # Plot additional lines for each feature in eng_features_list
        for i, eng_feature in enumerate(eng_features_list):
            eng_y_values = eng_feature[current_index]
            eng_x_values = list(range(len(eng_feature)))

            line_color = f"rgba(128, 128, 128, {0.2 + i * 0.01})"

            figure["data"].append(
                go.Scatter(
                    x=eng_x_values,
                    y=eng_y_values,
                    mode="lines",
                    name=f"Engineered Feature {i+1}",
                    line={"color": line_color},
                )
            )

        return figure


class SensorMapFigure:
    def __init__(self):
        self.dashboard_data = CustomDashboardData()
        self.sensor_info = self.dashboard_data.get_sensor_info()
        self.mapbox_access_token = open(
            get_mapbox_access_token_path(), encoding="utf8"
        ).read()
        self.fig = go.Figure()

    def create_map(self, lat, lon, name, values, color_scale, scale_name):
        colormap = blended_color_scale(scale_name, color_scale)
        colors = map_data_to_colors(values, colormap)
        self.fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=17,
                    opacity=0.7,
                    color=colors,
                ),
                text=name,
                hoverinfo="text",
                showlegend=False,
            )
        )

        self.fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            mapbox=dict(
                accesstoken=self.mapbox_access_token,
                bearing=0,
                center=dict(
                    lat=float(((lat.max() - lat.min()) / 2) + lat.min()),
                    lon=float(((lon.max() - lon.min()) / 2) + lon.min()),
                ),
                pitch=0,
                zoom=15,
                style="light",
            ),
        )
        return self.fig


class SensorMapCreator:
    def __init__(self):
        self.dashboard_data = CustomDashboardData()
        self.fig = SensorMapFigure()

    def completeness_values_to_floats(self):
        # Transform the completeness values to floats for plotting
        completeness_metrics = list(self.dashboard_data.get_completeness_metrics())
        completeness_values = [
            (float(value.strip("%")) / 100) for _, value in completeness_metrics
        ]

        return completeness_values

    def create_completeness_sensor_map_fig(self):
        lat, lon, name = self.dashboard_data.get_sensor_info()

        fig = self.fig.create_map(
            lat,
            lon,
            name,
            self.completeness_values_to_floats(),
            completeness_color_scale(),
            "Completeness",
        )
        return fig

    def freshness_values_to_floats(self):
        # Transform the freshness values to floats for plotting
        freshness_metrics = list(self.dashboard_data.get_freshness_metrics())
        values = [
            1
            / timedelta(
                days=int(value.split(" ")[0]),
                hours=int(value.split("\n")[1].split(":")[0]),
                minutes=int(value.split(":")[1]),
                seconds=int(value.split(":")[2].split(".")[0]),
            ).total_seconds()
            for _, value in freshness_metrics
        ]

        min_value = np.min(values)
        max_value = np.max(values)

        scaled_values = [
            (value - min_value) / (max_value - min_value) for value in values
        ]
        return scaled_values

    def create_freshness_sensor_map_fig(self):
        lat, lon, name = self.dashboard_data.get_sensor_info()

        fig = self.fig.create_map(
            lat,
            lon,
            name,
            self.freshness_values_to_floats(),
            freshness_color_scale(),
            "Freshness",
        )
        return fig
