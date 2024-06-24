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

from phd_project.src.utils.general_utils import (
    get_last_n_days_from_config,
    get_horizon_from_config,
    get_window_size_from_config,
)
from phd_project.src.utils.app_utils import find_tuple_by_first_element
from phd_project.apps.dashboard_data import (
    get_sensor_info,
    get_completeness_graph_data,
    get_latest_sensor_data,
    get_list_of_active_sensors,
    get_completeness_metrics,
    get_freshness_metrics,
    get_evaluation_predictions_data,
    get_training_windows_data,
)

first_sensor_in_list = get_list_of_active_sensors()[0]


def no_data_fig(sensor_name):
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


def create_completeness_metric(sensor_name=None):
    """
    Create a metric showing the completeness of the data.

    Args:
        sensor_name (str): The name of the sensor to create the metric for.

    Returns:
        str: A string containing the data completeness metric.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]
    data_list = get_completeness_metrics()
    completeness_value = find_tuple_by_first_element(data_list, sensor_name)
    if completeness_value is None:
        return "No data"
    return completeness_value


def create_freshness_metric(sensor_name=None):
    """
    Create a metric showing the freshness of the data.

    Args:
        sensor_name (str): The name of the sensor to create the metric for.

    Returns:
        str: A string containing the data freshness metric.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]
    data_list = get_freshness_metrics()
    freshness_value = find_tuple_by_first_element(data_list, sensor_name)
    if freshness_value is None:
        return "No data"
    return freshness_value


class SensorMapFigure:
    """
    A class to create a map figure.

    Attributes:
        mapbox_access_token (str): The mapbox access token.
        fig (Figure): A plotly figure.
    """

    def __init__(self, mapbox_access_token):
        self.mapbox_access_token = mapbox_access_token
        self.fig = go.Figure()

    def create_map(self, lat, lon, name):
        """
        Create a map showing sensor locations.
        """
        self.fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=dict(
                    size=17,
                    opacity=0.7,
                ),
                text=name,
                hoverinfo="text",
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

    def apply_colorbar(self, values, color_scale, colorbar_title="Value"):
        """
        Apply a colorbar to the map.
        """
        colorbar = dict(
            title=colorbar_title,
            titleside="top",
            tickmode="array",
            tickvals=[color_scale[i][0] for i in range(len(color_scale))],
            ticktext=[
                str(int(color_scale[i][0] * 100)) for i in range(len(color_scale))
            ],
            ticks="outside",
        )

        self.fig.update_traces(
            marker=dict(
                color=values,
                colorscale=color_scale,
                cmin=min(values),
                cmax=max(values),
                colorbar=colorbar,
            )
        )


def create_sensor_map_fig(
    values,
    colour_scale=None,
    colorbar_title="Value",
):
    """
    Create a map showing sensor locations.

    Args:
        values (list): A list of values to color the markers.
        colour_scale (list): A list of color values for the color scale.
        colorbar_title (str): The title of the colorbar.

    Returns:
        Figure: A plotly figure containing a map of sensor locations.
    """
    if colour_scale is None:
        colour_scale = [
            [0, "red"],
            [0.5, "yellow"],
            [1, "green"],
        ]
    mapbox_access_token = open("phd_project/apps/.mapbox_token", encoding="utf8").read()
    lat, lon, name = get_sensor_info()
    map_fig = SensorMapFigure(mapbox_access_token)
    map_fig.create_map(lat, lon, name)
    map_fig.apply_colorbar(values, colour_scale, colorbar_title)
    return map_fig.fig


def create_completeness_sensor_map_fig():
    """
    Create a map showing sensor locations colored by completeness.

    Returns:
        Figure: A plotly figure containing a map of sensor locations.
    """
    completeness_metrics = list(get_completeness_metrics())
    completeness_values = [
        (float(value.strip("%")) / 100) for _, value in completeness_metrics
    ]
    return create_sensor_map_fig(
        completeness_values, colorbar_title="Completeness Score"
    )


def create_freshness_sensor_map_fig():
    """
    Create a map showing sensor locations colored by freshness.

    Returns:
        Figure: A plotly figure containing a map of sensor locations.
    """
    freshness_metrics = list(get_freshness_metrics())
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

    # Calculate the minimum and maximum values
    min_value = np.min(values)
    max_value = np.max(values)

    scaled_values = [(value - min_value) / (max_value - min_value) for value in values]
    return create_sensor_map_fig(scaled_values, colorbar_title="Freshness Score")


def create_records_per_day_graph(sensor_name=None):
    """
    Create a graph showing missing data.

    Args:
        sensor_name (str): The name of the sensor to create the graph for.

    Returns:
        Figure: A plotly figure containing the missing data graph.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]

    data_list = get_completeness_graph_data()

    data = find_tuple_by_first_element(data_list, sensor_name)

    last_n_days = get_last_n_days_from_config()
    today = datetime.now()
    x_max = today.strftime("%Y-%m-%d")
    x_min = (today - timedelta(days=last_n_days)).strftime("%Y-%m-%d")

    if data is None:
        # Return a figure with a message indicating no data available
        return no_data_fig(sensor_name)
    assert isinstance(
        data, pd.DataFrame
    ), f"Expected data to be a pd.DataFrame, but got {type(data)}"

    missing_data_fig = go.Figure()
    missing_data_fig.add_trace(
        go.Bar(
            x=data["Timestamp"],
            y=data["Count"],
            name=sensor_name,
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


def create_latest_data_graph(sensor_name=None):
    """
    Create a graph showing the latest data.

    Args:
        sensor_name (str): The name of the sensor to create the graph for.

    Returns:
        Figure: A plotly figure containing the latest data graph.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]

    data_list = get_latest_sensor_data()
    data = find_tuple_by_first_element(data_list, sensor_name)

    if data is None:
        # Return a figure with a message indicating no data available
        return no_data_fig(sensor_name)
    assert isinstance(
        data, pd.DataFrame
    ), f"Expected data to be a pd.DataFrame, but got {type(data)}"

    latest_data_fig = go.Figure()
    latest_data_fig.add_trace(
        go.Scatter(
            x=data["Timestamp"],
            y=data["Value"],
            name=sensor_name,
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


def create_evaluation_graph(sensor_name=None):
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]

    data_list = get_evaluation_predictions_data()
    data = find_tuple_by_first_element(data_list, sensor_name, n_tuples=2)

    if data is None:
        # Return a figure with a message indicating no data available
        return no_data_fig(sensor_name)
    assert isinstance(data, tuple), f"Expected data to be a tuple, but got {type(data)}"

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


def create_training_windows_graph(sensor_name=None):
    """
    Create a graph showing training windows.

    Args:
        sensor_name (str): The name of the sensor to create the graph for.

    Returns:
        Figure: A plotly figure containing the training windows graph.
    """
    if sensor_name is None:
        sensor_name = get_list_of_active_sensors()[0]
    horizon = get_horizon_from_config()
    data = get_training_windows_data()

    if data is None or sensor_name not in data[3]:
        # Return a figure with a message indicating no data available
        return no_data_fig(sensor_name)
    assert isinstance(data, list), f"Expected data to be a list, but got {type(data)}"

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
                x=[len(x_values) + horizon], y=label_value, mode="markers", name="Label"
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


# if __name__ == "__main__":
