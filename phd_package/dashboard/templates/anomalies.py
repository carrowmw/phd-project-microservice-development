# dashboard/utils/templates/anomalies.py

from typing import Callable
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..data import CustomDashboardData


class TabTemplateAnomalies:
    def __init__(
        self,
        tabs_id: str,
        map_id: str,
        tab_label: str,
        tab_id: str,
        card_id: str,
        card_variable: str,
        graph_id: str,
        graph_func: Callable[[str], go.Figure],
        app: dash.Dash,
    ):
        self.app = app
        self.tabs_id = tabs_id
        self.map_id = map_id
        self.tab_label = tab_label
        self.tab_id = tab_id
        self.card_id = card_id
        self.card_variable = card_variable
        self.graph_id = graph_id
        self.graph = graph_func  # Store method reference

        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()

    def get_layout(self):
        return dbc.Row(
            id="anomaly_graph_row",
            children=[
                dbc.Card(
                    id = self.card_id,
                    children=dbc.CardBody(
                        [
                            html.H4(
                                children="Anomalies",
                                className="card-title",
                            ),
                            html.P(
                                id=self.card_variable,
                                className="card-text",
                            ),
                        ],
                        className="cardbody-element",
                    ),
                ),
                dcc.Graph(
                    id=self.graph_id,
                    className="graph-element",
                ),
            ],
            className="row-element",
        )

    def get_tab(self):
        return dbc.Tab(
            id="tab",
            label=self.tab_label,
            tab_id=self.tab_id,
            children=self.get_layout(),
            className="tab-element",
        )

    def setup_callbacks(self):
        @self.app.callback(
            [Output(self.card_variable, "children"),
            Output(self.graph_id, "figure"),
            ],
            [
                Input(self.tabs_id, "active_tab"),
                Input(self.map_id, "clickData"),
            ],
        )
        def update_tab(active_tab, map_click_data):
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"].split("<br>")[0]

            if active_tab == self.tab_id:
                total_anomalies = self.data.get_total_anomalies(sensor_name)
                card_text = f"Total anomalies: {total_anomalies}"
                graph = self.graph(sensor_name)
                return card_text, graph
            else:
                return dash.no_update
