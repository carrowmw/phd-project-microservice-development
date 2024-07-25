"""
Description: This module contains the main layout and callbacks for the sensor dashboard.
The dashboard is a web application that displays information about sensors in a network.
The dashboard is divided into two main sections: the left section displays a map overview
of the sensors, while the right section displays detailed information about the selected sensor.

The dashboard consists of the following components:
- A title bar at the top of the page.
- A row of tabs on the left side of the page for selecting different views.
- A row of tabs on the right side of the page for displaying detailed information.
- A map overview tab that displays a map of the sensor network.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

from dashboard.graphs import (
    SensorMapFigure,
    SensorMapCreator,
    PipelineChartCreator,
)

from dashboard.data import CustomDashboardData

from utils.config_helper import get_horizon

from pipeline.__main__ import Pipeline


class SensorDashboardApp:
    """ """

    def __init__(self):
        self.pipeline = Pipeline()
        self.chart_figures = PipelineChartCreator()
        self.sensor_map = SensorMapCreator()
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, "/assets/custom_styles.css"],
        )
        self.setup_layout()
        self.setup_callbacks()

    def run_pipeline(self):
        self.pipeline.run_pipeline()

    def setup_layout(self):
        overview_tab = dbc.Tab(
            label="Data Overview",
            id="data_overview_tab",
            children=[
                dbc.Row(
                    dcc.Graph(
                        id="records_per_day_graph",
                        figure=self.chart_figures.create_records_per_day_graph(),
                        className="graph-element records_per_day_graph",
                    ),
                    className="row-element",
                ),
                dbc.Row(
                    dcc.Graph(
                        id="graph_b",
                        figure=create_latest_data_graph(),
                        className="graph-element graph_b",
                    ),
                    className="row-element",
                ),
            ],
            className="tab-element",  # Uses the same formatting as the tabs
        )

        right_tabs_column = dbc.Col(
            id="right_tabs_column",
            children=[
                dbc.Tabs(
                    id="right_tabs",
                    children=[
                        overview_tab,
                        # performance_tab,
                        # predictions_tab,
                        # preprocessing_tab,
                        # engineering_tab,
                        # training_windows_tab,
                    ],
                    className="tabs-element",
                )
            ],
            width=6,
            className="column-element",
        )

        left_column = dbc.Col(
            children=[
                map_overview_column,
            ],
            width=6,
            className="column-element",
        )

        right_title_column = dbc.Col(
            children=[
                html.Br(),
                html.H5(
                    id="selected-sensor-name",
                    className="sensor-name",
                    style={"text-align": "center"},
                ),
                html.Br(),
            ],
            width=6,
            align="center",
            className="column-element",
        )

        left_title_column = dbc.Col(
            children=[
                html.H2(
                    "Sensor Dashboard",
                    style={"text-align": "center"},
                ),
            ],
            width=6,
            align="center",
            className="column-element",
        )

        top_row = dbc.Row(
            children=[
                html.Br(),
                left_title_column,
                right_title_column,
            ],
            align="center",
            className="row-element",
        )

        bottom_row = dbc.Row(
            children=[left_column, right_tabs_column],
            align="center",
            className="row-element",
        )

        self.app.layout = dbc.Container(
            children=[top_row, bottom_row],
            fluid=True,
            className="container-element",
        )

    preprocessing_tab = dbc.Tab(
        label="Preprocessing",
        tab_id="preprocessing_tab",
        children=[
            dbc.Row(
                dash_table.DataTable(
                    id="table_a",
                    columns=[
                        {"name": i, "id": i} for i in get_preprocessing_table().columns
                    ],
                    data=get_preprocessing_table().to_dict("records"),
                    style_table={
                        "height": "50vh",
                        "width": "25vw",
                        "margin": "50px auto",
                        "overflowY": "auto",
                    },
                    style_cell={"textAlign": "left", "padding": "5px"},
                ),
                className="row-element",
            ),
        ],
        className="tab-element",
    )

    performance_tab = dbc.Tab(
        label="Performance",
        tab_id="performance_tab",
        children=[
            dbc.Row(
                dcc.Graph(
                    id="graph_c",
                    figure={},
                    className="graph-element graph_c",
                ),
                className="row-element",
            ),
            dbc.Row(
                dcc.Graph(
                    id="graph_d",
                    figure={},
                    className="graph-element graph_d",
                ),
                className="row-element",
            ),
        ],
        className="tab-element",
    )

    predictions_tab = dbc.Tab(
        label="Predictions",
        tab_id="predictions_tab",
        children=[
            dbc.Row(
                dcc.Graph(
                    id="graph_e",
                    figure={},
                    className="graph-element graph_e",
                ),
                className="row-element",
            ),
            dbc.Row(
                dcc.Graph(
                    id="graph_f",
                    figure={},
                    className="graph-element graph_f",
                ),
                className="row-element",
            ),
        ],
        className="tab-element",
    )

    training_windows_tab = dbc.Tab(
        label="Training Windows",
        tab_id="training_windows_tab",
        children=[
            dbc.Row(
                dcc.Graph(
                    id="graph_g",
                    figure=create_training_windows_graph(),
                    className="graph-element graph_g",
                ),
                className="row-element",
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        html.Button(
                            "←",
                            id="prev-window-button",
                            n_clicks=0,
                        ),
                    ),
                    dbc.Col(
                        html.Div(
                            id="frame-counter",
                            children="1/1",
                        ),
                    ),
                    dbc.Col(
                        html.Button(
                            "→",
                            id="next-window-button",
                            n_clicks=0,
                        ),
                    ),
                ],
            ),
        ],
        className="tab-element",
    )

    map_overview_column = dbc.Col(
        id="map_overview_column",
        children=[
            dbc.Row(
                dcc.Dropdown(
                    id="layer_selector",
                    options=[
                        {"label": "Completeness", "value": "completeness"},
                        {"label": "Freshness", "value": "freshness"},
                    ],
                    value="completeness",
                    className="dropdown-element",
                ),
            ),
            dbc.Row(
                dcc.Graph(
                    id="map_overview_figure",
                    figure={},
                    className="graph-element map_overview_figure",
                ),
                className="row-element",
            ),
            dbc.Row(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(
                                "Data Metric",
                                className="card-title",
                            ),
                            html.P(
                                id="data-metric-value",
                                className="card-text",
                            ),
                        ],
                        className="card-element",
                    ),
                    className="card-element",
                ),
            ),
        ],
        class_name="column-element",
        style={"width": "100%"},
    )

    right_tabs_column = dbc.Col(
        id="right_tabs_column",
        children=[
            dbc.Tabs(
                id="right_tabs",
                children=[
                    overview_tab,
                    performance_tab,
                    predictions_tab,
                    preprocessing_tab,
                    engineering_tab,
                    training_windows_tab,
                ],
                className="tabs-element",
            )
        ],
        width=6,
        className="column-element",
    )

    left_column = dbc.Col(
        children=[
            map_overview_column,
        ],
        width=6,
        className="column-element",
    )

    right_title_column = dbc.Col(
        children=[
            html.Br(),
            html.H5(
                id="selected-sensor-name",
                className="sensor-name",
                style={"text-align": "center"},
            ),
            html.Br(),
        ],
        width=6,
        align="center",
        className="column-element",
    )

    left_title_column = dbc.Col(
        children=[
            html.H2(
                "Sensor Dashboard",
                style={"text-align": "center"},
            ),
        ],
        width=6,
        align="center",
        className="column-element",
    )

    top_row = dbc.Row(
        children=[
            html.Br(),
            left_title_column,
            right_title_column,
        ],
        align="center",
        className="row-element",
    )

    bottom_row = dbc.Row(
        children=[left_column, right_tabs_column],
        align="center",
        className="row-element",
    )

    @app.callback(
        Output("map_overview_figure", "figure"),
        Input("layer_selector", "value"),
    )
    def update_map(selected_layer):
        if selected_layer == "completeness":
            map_fig = create_completeness_sensor_map_fig()
        elif selected_layer == "freshness":
            map_fig = create_freshness_sensor_map_fig()
        else:
            map_fig = create_completeness_sensor_map_fig()

        return map_fig

    @app.callback(
        Output("data-metric-value", "children"),
        Input("map_overview_figure", "clickData"),
        Input("layer_selector", "value"),
    )
    def update_data_metric_value(map_clickData, selected_layer):
        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]

        if selected_layer == "completeness":
            metric_value = create_completeness_metric(sensor_name)
        elif selected_layer == "freshness":
            metric_value = create_freshness_metric(sensor_name)
        else:
            metric_value = create_completeness_metric(sensor_name)

        return metric_value

    @app.callback(
        Output("selected-sensor-name", "children"),
        Input("map_overview_figure", "clickData"),
    )
    def update_selected_sensor_name(map_clickData):

        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]
        return f"{sensor_name}"

    @app.callback(
        Output("graph_a", "figure"),
        Output("graph_b", "figure"),
        Input("map_overview_figure", "clickData"),
    )
    def update_overview_tab(map_clickData):
        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]

        missing_data_fig = create_records_per_day_graph(sensor_name)
        latest_data_fig = create_latest_data_graph(sensor_name)

        return missing_data_fig, latest_data_fig

    @app.callback(
        Output("graph_c", "figure"),
        Output("graph_d", "figure"),
        Input("map_overview_figure", "clickData"),
        Input("right_tabs", "active_tab"),
    )
    def update_performance_tab(map_clickData, right_tabs_column):
        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]
        if right_tabs_column == "performance_tab":
            evaluation_fig = create_evaluation_graph(sensor_name)
            # Create and return figures for graph_c and graph_d
            return evaluation_fig, dash.no_update
        else:
            return dash.no_update, dash.no_update

    @app.callback(
        Output("graph_e", "figure"),
        Output("graph_f", "figure"),
        Input("map_overview_figure", "clickData"),
        Input("right_tabs", "active_tab"),
    )
    def update_predictions_tab(map_clickData, right_tabs_column):
        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]

        if right_tabs_column == "predictions_tab":
            # Create and return figures for graph_e and graph_f
            return dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update

    @app.callback(
        Output("table_a", "data"),
        Input("map_overview_figure", "clickData"),
        Input("right_tabs", "active_tab"),
    )
    def update_preprocessing_tab(map_clickData, right_tabs_column):

        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]

        if right_tabs_column == "preprocessing_tab":
            # Update and return the data for table_a based on the selected date range
            data = get_preprocessing_table(sensor_name)
            return data.to_dict("records")
        else:
            return dash.no_update

    # Update the callback function to handle the new graph and button clicks
    @app.callback(
        Output("graph_g", "figure"),
        Output("frame-counter", "children"),
        Input("map_overview_figure", "clickData"),
        Input("right_tabs", "active_tab"),
        Input("prev-window-button", "n_clicks"),
        Input("next-window-button", "n_clicks"),
        State("frame-counter", "children"),
    )
    def update_training_windows_tab(
        map_clickData, right_tabs_column, prev_clicks, next_clicks, frame_counter
    ):
        if map_clickData is None:
            sensor_name = get_random_sensor()
        else:
            sensor_name = map_clickData["points"][0]["text"]

        if right_tabs_column == "training_windows_tab":
            data = get_training_windows_data()
            sensor_index = data[3].index(sensor_name)
            features = data[0][sensor_index]
            labels = data[1][sensor_index]
            num_items = len(features)

            ctx = dash.callback_context
            if not ctx.triggered:
                current_frame = 1
                n_clicks = 0
            else:
                current_frame = int(frame_counter.split("/")[0])
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]

                if button_id == "next-window-button":
                    n_clicks = 1
                elif button_id == "prev-window-button":
                    n_clicks = -1
                else:
                    n_clicks = 0

            current_index = (current_frame + n_clicks - 1) % num_items
            if current_index == -1:
                current_index = num_items - 1

            x_values = list(range(len(features[current_index])))
            y_values = list(features[current_index])
            label_value = labels[current_index]
            horizon = get_horizon()

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
                    "xaxis": {"title": "X-axis"},
                    "yaxis": {"title": "Y-axis"},
                },
            }

            new_frame_counter = f"{current_frame + n_clicks}/{num_items}"
            return figure, new_frame_counter
        else:
            return dash.no_update, dash.no_update

    return app
