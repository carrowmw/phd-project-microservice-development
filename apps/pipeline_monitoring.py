"""
This module contains the Flask and Dash application for visualizing the training data
from the pipeline.
"""

import os
import logging
from typing import Any, Dict, List, Tuple

import dash
import numpy as np
import plotly.graph_objs as go
import torch
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Flask

from src.pipeline import (
    apply_feature_engineering,
    load_data,
    preprocess_raw_data,
)
from src.utils.app_utils import (
    create_and_load_file_path,
    load_data_from_file,
    save_data_to_file,
)

from src.utils.fstore_utils import (
    create_raw_file_path_from_config,
    create_dataloaders_file_path_from_config,
    download_raw_data,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloaders,
)

from src.utils.general_utils import get_window_size_from_config, get_horizon_from_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WINDOW_SIZE = get_window_size_from_config()
HORIZON = get_horizon_from_config()
APP_DATA_DIRECTORY = "sliding_windows"
API_CONFIG_FILE_PATH = "configs/api_config.json"


def retrieve_pipeline_monitoring_app_data() -> List[List[np.ndarray]]:
    """
    Retrieves and formats application data for visualization in a missing data app.

    Returns:
        List[List[np.ndarray]]: A list of lists containing the training windows and labels for the LSTM.
    """
    file_path = create_and_load_file_path(API_CONFIG_FILE_PATH, APP_DATA_DIRECTORY)
    app_data = load_data_from_file(file_path)

    if app_data is not None:
        logger.info("App data loaded from file.")
        return app_data

    logger.info("Executing pipeline steps to retrieve app data.")
    if torch.backends.mps.is_available():
        print("MPS backend available")
    else:
        print("Warning, cannot move to MPS backend")

    # Checkpoint 1: Check if raw data exists
    raw_data_path = create_raw_file_path_from_config()
    if os.path.exists(raw_data_path):
        print("Preprocessed data found. Skipping preprocessing step.")
        raw_dfs = load_raw_data()
    else:
        print("Preprocessed data not found. Running preprocessing step.")
        raw_dfs = download_raw_data()
        save_raw_data(raw_dfs, raw_data_path)

    # Checkpoint 2: Check if preprocessed data exists
    preprocessed_data_path = create_raw_file_path_from_config().replace(
        "raw", "preprocessed"
    )
    if os.path.exists(preprocessed_data_path):
        print("Preprocessed data found. Skipping preprocessing step.")
        preprocessed_dfs = load_preprocessed_data()
    else:
        print("Preprocessed data not found. Running preprocessing step.")
        preprocessed_dfs = preprocess_raw_data(raw_dfs)
        save_preprocessed_data(preprocessed_dfs, preprocessed_data_path)

    # Checkpoint 3: Check if engineered features exist
    engineered_data_path = create_raw_file_path_from_config().replace(
        "raw", "engineered"
    )
    if os.path.exists(engineered_data_path):
        print("Engineered features found. Skipping feature engineering step.")
        engineered_dfs = load_engineered_data()
    else:
        print("Engineered features not found. Running feature engineering step.")
        engineered_dfs = apply_feature_engineering(preprocessed_dfs)
        save_engineered_data(engineered_dfs, engineered_data_path)

    # Checkpoint 4: Check if data loaders exist
    data_loaders_path = create_dataloaders_file_path_from_config()
    if os.path.exists(data_loaders_path):
        print("Data loaders found. Skipping data loading step.")
        data_loaders_list = load_dataloaders()
    else:
        print("Data loaders not found. Running data loading step.")
        data_loaders_list = load_data(engineered_dfs)
        save_dataloaders(data_loaders_list, data_loaders_path)

    print("Unbatching...")
    app_data = [[], [], [], []]
    for _, (_, _, _, val_dataloader, _) in enumerate(data_loaders_list):
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

    for pipeline_object in data_loaders_list:
        sensor_name = pipeline_object[0]
        app_data[3].append(sensor_name)
    print("Unbatching complete")
    save_data_to_file(file_path, app_data)
    logger.info("App data saved to file.")

    return app_data


def create_graph_layout(sensor_index: int) -> Dict[str, Any]:
    """
    Creates a layout dictionary for a graph.

    Args:
        sensor_index (int): The index of the sensor for which the graph is being created.

    Returns:
        Dict[str, Any]: A dictionary representing the layout of the graph.
    """
    return {
        "data": [
            go.Scatter(x=[], y=[], mode="lines", name="Training Window"),
            go.Scatter(x=[], y=[], mode="markers", name="Label"),
        ],
        "layout": {
            "title": {
                "text": f"Sensor {sensor_index + 1}",
                "font": {"size": 24},
                "x": 0.5,
            },
            "xaxis": {"title": "X-axis"},
            "yaxis": {"title": "Y-axis"},
        },
    }


def update_graph(
    n_clicks: int,
    current_frame: int,
    num_items: int,
    sensor_index: int,
    window_size: int,
    horizon: int,
) -> Tuple[Dict[str, Any], str]:
    """
    Updates the graph with the next or previous training window and label.

    Args:
        n_clicks (int): The number of clicks on the button (1 for next, -1 for previous).
        current_frame (int): The current frame number.
        num_items (int): The total number of items (training windows and labels).
        sensor_index (int): The index of the sensor for which the graph is being updated.
        window_size (int): The size of the sliding window used for training.
        horizon (int): The size of the horizon used for training

    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing the updated graph figure and frame counter.
    """
    try:
        features = app_data[0][sensor_index]
        labels = app_data[1][sensor_index]
        eng_features_list = app_data[2][sensor_index]
        sensor_name = app_data[3][sensor_index]

        current_frame = (current_frame + n_clicks) % num_items
        if current_frame == 0:
            current_frame = num_items

        current_index = current_frame - 1  # * window_size

        x_values = list(range(len(features[current_index])))
        y_values = list(features[current_index])
        label_value = labels[current_index]
        figure = {
            "data": [
                go.Scatter(
                    x=x_values, y=y_values, mode="lines", name="Training Window"
                ),
                go.Scatter(
                    x=[len(x_values) + horizon],
                    y=label_value,
                    mode="markers",
                    name="Label",
                ),
            ],
            "layout": {
                "title": {"text": f"{sensor_name}", "font": {"size": 24}, "x": 0.5},
                "xaxis": {"title": "X-axis"},
                "yaxis": {"title": "Y-axis"},
            },
        }
        # Plot additional lines for each feature in eng_features_list
        for i, eng_feature in enumerate(eng_features_list):
            eng_y_values = eng_feature[current_index]
            eng_x_values = list(range(len(eng_feature)))

            if i == 1:
                # Set the color to pink for the feature at index 2
                line_color = "pink"
            else:
                # Keep the original color for other features
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

        frame_counter = f"{current_frame}/{num_items}"
        return figure, frame_counter

    except Exception as e:
        logger.error(f"Error updating graph: {str(e)}")
        return {}, "0/0"


app_data = retrieve_pipeline_monitoring_app_data()

server = Flask(__name__)
app = dash.Dash(__name__, server=server, routes_pathname_prefix="/dashboard/")

graph_divs = []
for sensor_index, _ in enumerate(app_data[0]):
    num_items = len(app_data[0][sensor_index])
    graph_divs.append(
        html.Div(
            children=[
                dcc.Graph(
                    id=f"graph-{sensor_index}",
                    figure=create_graph_layout(sensor_index),
                    style={"width": "100%"},
                ),
                html.Div(
                    [
                        html.Button(
                            "←",
                            id=f"prev-window-button-{sensor_index}",
                            n_clicks=0,
                            style={"margin": "0 10px"},
                        ),
                        html.Div(
                            id=f"frame-counter-{sensor_index}",
                            children=f"1/{num_items}",
                            style={
                                "font-size": "24px",
                                "font-family": "Arial, sans-serif",
                                "text-align": "center",
                                "margin": "0 20px",
                            },
                        ),
                        html.Button(
                            "→",
                            id=f"next-window-button-{sensor_index}",
                            n_clicks=0,
                            style={"margin": "0 10px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                    },
                ),
            ],
            id=f"section-{sensor_index}",
            style={
                "padding": "20px",
                "border": "1px solid #ddd",
                "height": "500px",
                "width": "100%",
            },
        )
    )

app.layout = html.Div(
    children=[
        html.Div(
            id="graphs-container",
            children=graph_divs,
            style={"maxWidth": "2400", "margin": "auto", "textAlign": "center"},
        ),
    ]
)

for sensor_index, _ in enumerate(app_data[0]):
    num_items = len(app_data[0][sensor_index])  # // WINDOW_SIZE

    @app.callback(
        [
            Output(f"graph-{sensor_index}", "figure"),
            Output(f"frame-counter-{sensor_index}", "children"),
        ],
        [
            Input(f"next-window-button-{sensor_index}", "n_clicks"),
            Input(f"prev-window-button-{sensor_index}", "n_clicks"),
        ],
        [State(f"frame-counter-{sensor_index}", "children")],
    )
    def update_graph_callback(
        next_clicks: int,
        prev_clicks: int,
        frame_counter: str,
        sensor_index: int = sensor_index,
        num_items: int = num_items,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Callback function to update the graph when the button is clicked.

        Args:
            next_clicks (int): The number of clicks on the "Next" button.
            prev_clicks (int): The number of clicks on the "Previous" button.
            frame_counter (str): The current frame counter value.
            sensor_index (int, optional): The index of the sensor for which the graph is being updated.
            num_items (int, optional): The total number of items (training windows and labels).

        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing the updated graph figure and frame counter.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            current_frame = 1
            n_clicks = 0
        else:
            current_frame = int(frame_counter.split("/")[0])
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == f"next-window-button-{sensor_index}":
                n_clicks = 1
            elif button_id == f"prev-window-button-{sensor_index}":
                n_clicks = -1
            else:
                n_clicks = 0

        return update_graph(
            n_clicks, current_frame, num_items, sensor_index, WINDOW_SIZE, HORIZON
        )


if __name__ == "__main__":
    app.run_server(debug=True)
