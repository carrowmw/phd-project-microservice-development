# dashboard/graphs.py

"""
This module contains functions for creating the graphs and metrics that are displayed on the dashboard.

It calls functions from the `dashboard_data` module to fetch the data required for the graphs and metrics.

It contains the following functions:
- `create_completeness_metric`: Creates a metric showing the completeness of the data.
- `create_map_fig`: Creates a map showing sensor locations.
- `create_missing_data_graph`: Creates a graph showing missing data.
- `create_latest_data_graph`: Creates a graph showing the latest data.
- `create_test_graph`: Creates a graph showing test predictions.
- `create_training_windows_graph`: Creates a graph showing training windows.
"""


import pandas as pd
import plotly.graph_objects as go

from ..config.paths import get_mapbox_access_token_path

from ..utils.config_helper import get_n_days, get_query_agnostic_start_and_end_date
from ..utils.data_helper import load_test_metrics, find_tuple_by_first_element

from .data import CustomDashboardData
from .utils.color_helper import (
    base_colors,
    category_colors,
    completeness_color_scale,
    freshness_color_scale,
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
        self.test_predictions = self.dashboard_data.get_test_predictions()
        self.train_metrics = self.dashboard_data.get_train_metrics()
        self.training_windows = self.dashboard_data.get_training_windows()
        self.sensor_info = self.dashboard_data.get_sensor_info()
        self.completeness_graph_data = self.dashboard_data.get_completeness_graph_data()
        self.first_sensor_in_list = self.dashboard_data.active_sensors[0]
        self.xmin, self.xmax = get_query_agnostic_start_and_end_date()
        self.n_days = get_n_days()
        self.test_dummy = load_test_metrics()
        self.category_colors = list(category_colors.values())

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
        # data_list = self.completeness_metrics
        # print(f"TEST: self.completeness_metrics: {self.completeness_metrics}")
        completeness_value = self.completeness_metrics.loc[
            self.completeness_metrics["sensor_name"] == sensor_name, "string"
        ].values[0]
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
        freshness_value = self.freshness_metrics.loc[
            self.freshness_metrics["sensor_name"] == sensor_name, "string"
        ].values[0]
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
                range=[self.xmin, self.xmax],  # Set the x-axis range
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

    def create_test_predictions_graph(self, sensor_name=None) -> go.Figure:
        """
        Create a graph showing test predictions.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the test predictions graph.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data = find_tuple_by_first_element(
            self.test_predictions, sensor_name, n_tuples=2
        )
        if data is None:
            return self.no_data_fig(sensor_name)
        assert isinstance(
            data, tuple
        ), f"Expected data to be a tuple, but got {type(data)}"

        predictions = [item[0] for item in data[0]]
        labels = [item[0] for item in data[1]]

        test_fig = go.Figure()
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(labels))),
                y=labels,
                mode="lines",
                name="Labels",
                line=dict(color=self.category_colors[0]),
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(labels))),
                y=predictions,
                mode="lines",
                name="Predictions",
                line=dict(color=self.category_colors[1]),
            )
        )

        test_fig.update_layout(
            title={
                "text": "Test Predictions",
                "font": {"size": 20},
                "x": 0.5,  # Center the title horizontally
            },
            xaxis=dict(
                title="Value",
            ),
            yaxis=dict(
                title="Time",
                # range=[0, 1],
            ),
        )

        return test_fig

    def create_train_metrics_graph(self, sensor_name=None) -> go.Figure:
        """
        Create a graph showing train metrics.

        Args:
            sensor_name (str): The name of the sensor to create the graph for.

        Returns:
            Figure: A plotly figure containing the  train metrics graph.
        """
        if sensor_name is None:
            sensor_name = self.first_sensor_in_list
        data = find_tuple_by_first_element(self.train_metrics, sensor_name, n_tuples=2)
        if data is None:
            return self.no_data_fig(sensor_name)
        assert isinstance(
            data, tuple
        ), f"Expected data to be a tuple, but got {type(data)}"

        validation_metrics = data[1]

        test_fig = go.Figure()
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(validation_metrics))),
                y=[item["Val loss"] for item in validation_metrics],
                mode="lines+markers",
                name="Val loss",
                line=dict(color=self.category_colors[0]),
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(validation_metrics))),
                y=[item["Val MAPE"] for item in validation_metrics],
                mode="lines+markers",
                name="Val MAPE",
                line=dict(color=self.category_colors[1]),
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(validation_metrics))),
                y=[item["Val RMSE"] for item in validation_metrics],
                mode="lines+markers",
                name="Val RMSE",
                line=dict(color=self.category_colors[2]),
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(validation_metrics))),
                y=[item["Val R2"] for item in validation_metrics],
                mode="lines+markers",
                name="Val R2",
                line=dict(color=self.category_colors[3]),
            )
        )

        test_fig.update_layout(
            title={
                "text": "Train Metrics",
                "font": {"size": 20},
                "x": 0.5,  # Center the title horizontally
            },
            xaxis=dict(
                title="Epoch",
            ),
            yaxis=dict(
                title="Metric",
            ),
        )

        return test_fig


class SensorMapFigure:
    def __init__(self, metrics=None):
        self.dashboard_data = CustomDashboardData()
        self.sensor_info = self.dashboard_data.get_sensor_info()
        self.metrics = metrics
        self.mapbox_access_token = open(
            get_mapbox_access_token_path(), encoding="utf8"
        ).read()
        self.fig = go.Figure()

    def create_map(self, lat, lon, name, values, colorscale):
        self.fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=17,
                    opacity=0.7,
                    color=values,
                    colorscale=colorscale,
                    cmin=0,
                    cmax=1,
                ),
                text=name,
                hoverinfo="text",
                showlegend=False,
                hovertemplate="<i>%{text}<i><br>"
                + "<b>Metric</b>: %{marker.color:.2f}",
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
        self.completeness_metrics = self.dashboard_data.get_completeness_metrics()
        self.freshness_metrics = self.dashboard_data.get_freshness_metrics()
        self.fig = SensorMapFigure()

    def create_completeness_sensor_map_fig(self):
        lat, lon, name = self.dashboard_data.get_sensor_info()

        fig = self.fig.create_map(
            lat,
            lon,
            name,
            values=self.completeness_metrics["float"],
            colorscale=completeness_color_scale(),
        )
        return fig

    def create_freshness_sensor_map_fig(self):
        lat, lon, name = self.dashboard_data.get_sensor_info()
        print(
            f"TEST: self.freshness_metrics['norm_log']: {self.freshness_metrics['norm_log']}"
        )
        fig = self.fig.create_map(
            lat,
            lon,
            name,
            values=self.freshness_metrics["norm_log"],
            colorscale=freshness_color_scale(),
        )
        return fig
