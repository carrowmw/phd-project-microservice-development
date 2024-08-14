# dashboard/utils/templates/double_graph.py

from typing import Callable
import dash
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..data import CustomDashboardData


class TabTemplateDoubleGraph:
    def __init__(
        self,
        tabs_id: str,
        map_id: str,
        tab_label: str,
        tab_id: str,
        graph_id_a: str,
        graph_a_func: Callable[[str], go.Figure],
        graph_id_b: str,
        graph_b_func: Callable[[str], go.Figure],
        app: dash.Dash,
    ):
        self.app = app
        self.tabs_id = tabs_id
        self.map_id = map_id
        self.tab_label = tab_label
        self.tab_id = tab_id
        self.graph_id_a = graph_id_a
        self.graph_a = graph_a_func  # Store method reference
        self.graph_id_b = graph_id_b
        self.graph_b = graph_b_func  # Store method reference
        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()

    def get_layout(self):
        return dbc.Row(
            id="double_graph_row",
            children=[
                dcc.Graph(
                    id=self.graph_id_a,
                    className="graph-element double_graph",
                ),
                dcc.Graph(
                    id=self.graph_id_b,
                    className="graph-element double_graph",
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
            [Output(self.graph_id_a, "figure"), Output(self.graph_id_b, "figure")],
            [
                Input(self.tabs_id, "active_tab"),
                Input(self.map_id, "clickData"),
            ],
        )
        def update_tab(active_tab, map_click_data):
            print("TabTemplateDoubleGraph.setup_callbacks()")
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"]

            if active_tab == self.tab_id:
                graph_a = self.graph_a(sensor_name)
                graph_b = self.graph_b(sensor_name)
                return graph_a, graph_b
            else:
                return dash.no_update, dash.no_update
