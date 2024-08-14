# dashboard/utils/templates/dataloader_viewer.py

from typing import Callable
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ...utils.config_helper import get_horizon
from ..data import CustomDashboardData
from ..utils.color_helper import base_colors


class TabTemplateDataLoaderViewer:
    def __init__(
        self,
        tabs_id: str,
        map_id: str,
        tab_label: str,
        tab_id: str,
        graph_id: str,
        graph_func: Callable[[str], go.Figure],
        app: dash.Dash,
    ):
        self.app = app
        self.tabs_id = tabs_id
        self.map_id = map_id
        self.tab_label = tab_label
        self.tab_id = tab_id
        self.graph_id = graph_id
        self.graph = graph_func  # Store method reference
        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()

    def get_layout(self):
        return dbc.Row(
            id="graph_row",
            children=[
                dcc.Store(id="prev-sensor-name"),
                dcc.Graph(
                    id=self.graph_id,
                    className="graph-element",
                ),
                html.Div(
                    children=[
                        dbc.Button(
                            id="prev-window-button",
                            children="Previous",
                            n_clicks=0,
                            color=base_colors["Mid Blue"],
                            className="button-element",
                        ),
                        html.Span(
                            id="frame-counter",
                            children="Sequence 1 out of 1",
                            className="frame-counter-element",
                            style={"verticalAlign": "middle"},
                        ),
                        dbc.Button(
                            id="next-window-button",
                            children="Next",
                            n_clicks=0,
                            color=base_colors["Mid Blue"],
                            className="button-element",
                        ),
                    ],
                    className="button-frame-container",  # Apply Flexbox here
                ),
            ],
            className="row-element",
        )

    def get_tab(self):
        return dbc.Tab(
            label=self.tab_label,
            tab_id=self.tab_id,
            children=self.get_layout(),
            className="tab-element",
        )

    def setup_callbacks(self):
        @self.app.callback(
            [
                Output(self.graph_id, "figure"),
                Output("frame-counter", "children"),
                Output(
                    "prev-sensor-name", "data"
                ),  # This updates the dcc.Store component
            ],
            [
                Input(self.map_id, "clickData"),
                Input(self.tabs_id, "active_tab"),
                Input("prev-window-button", "n_clicks"),
                Input("next-window-button", "n_clicks"),
            ],
            [
                State("frame-counter", "children"),  # Store the current frame counter
                State("prev-sensor-name", "data"),  # Store the previous sensor name
            ],
        )
        def update_tab(
            map_click_data,
            active_tab,
            prev_clicks,
            next_clicks,
            frame_counter,
            prev_sensor_name,
        ):
            print("TabTemplateDataLoaderViewer.setup_callbacks()")
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"]

            if active_tab == self.tab_id:
                # Get the number of frames from the data
                data = self.data.get_training_windows()
                sensor_index = data[3].index(sensor_name)
                input_feature = data[0][sensor_index]
                labels = data[1][sensor_index]
                eng_features = data[2][sensor_index]
                number_of_items = len(input_feature)

                if sensor_name != prev_sensor_name:
                    current_frame = 1
                    number_of_clicks = 0
                else:

                    current_frame = int(frame_counter.split(" ")[1])
                    callback_context = dash.ctx
                    if not callback_context.triggered:
                        # current_frame = 1
                        number_of_clicks = 0
                    else:
                        button_id = callback_context.triggered[0]["prop_id"].split(".")[
                            0
                        ]
                        if button_id == "next-window-button":
                            number_of_clicks = 1
                        elif button_id == "prev-window-button":
                            number_of_clicks = -1
                        else:
                            number_of_clicks = 0

                current_index = (current_frame + number_of_clicks - 1) % number_of_items
                if current_index == -1:
                    current_index = number_of_items - 1
                new_frame_counter = (
                    f"Sequence {current_index + 1} out of {number_of_items}"
                )

                # Create the data for the current frame
                x_values = list(range(len(input_feature[current_index])))
                y_values = list(input_feature[current_index])
                label_value = labels[current_index]
                horizon = get_horizon()

                # Create the graph for the current frame
                fig = {
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
                        "xaxis": {"title": "X-axis"},
                        "yaxis": {"title": "Y-axis", "range": [-1.5, 6]},
                    },
                }
                # Plot additional lines for each feature in eng_features_list
                for i, eng_feature in enumerate(eng_features):
                    eng_x_values = list(range(len(eng_feature[current_index])))
                    eng_y_values = list(eng_feature[current_index])

                    line_color = f"rgba(128, 128, 128, {0.1 + i * 0.05})"

                    fig["data"].append(
                        go.Scatter(
                            x=eng_x_values,
                            y=eng_y_values,
                            mode="lines",
                            name=f"Engineered Feature {i+1}",
                            line={"color": line_color},
                        )
                    )
                # graph = self.graph(sensor_name, frame_counter)
                return fig, new_frame_counter, sensor_name
            else:
                return dash.no_update, dash.no_update, dash.no_update
