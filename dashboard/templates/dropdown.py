# dashboard/utils/templates/dropdown.py

from typing import Callable
import dash
from dash import dcc
from dash.dependencies import Input, Output


from dashboard.data import CustomDashboardData
from dashboard.utils.color_helper import (
    completeness_color_scale,
    freshness_color_scale,
    plot_color_gradient,
)


class DropdownTemplate:
    def __init__(
        self,
        dropdown_id: str,
        map_id: str,
        legend_id: str,
        map_object: Callable[[str], dict],
        app: dash.Dash,
    ):
        self.dropdown_id = dropdown_id
        self.map_id = map_id
        self.legend_id = legend_id
        self.map_object = map_object
        self.app = app
        self.data = CustomDashboardData()

    def get_layout(self):
        return dcc.Dropdown(
            id=self.dropdown_id,
            options=[
                {"label": "Completeness - Availability", "value": "completeness"},
                {"label": "Timeliness - Freshness", "value": "freshness"},
            ],
            value="completeness",
            className="dropdown-element",
        )

    def setup_callbacks(self):
        @self.app.callback(
            [Output(self.map_id, "figure"), Output(self.legend_id, "figure")],
            Input(self.dropdown_id, "value"),
        )
        def update_map(selected_layer):
            print(
                "DropdownTemplate.setup_callbacks() - Layer selected:", selected_layer
            )
            if selected_layer == "completeness":
                colorscale = completeness_color_scale()
                legend = plot_color_gradient(selected_layer, colorscale)
                map_fig = self.map_object.create_completeness_sensor_map_fig()
                return map_fig, legend
            elif selected_layer == "freshness":
                tick_text = self.data.get_freshness_legend_tick_text()
                colorscale = freshness_color_scale()
                legend = plot_color_gradient(selected_layer, colorscale, tick_text)
                map_fig = self.map_object.create_freshness_sensor_map_fig()
                return map_fig, legend
            else:
                return (
                    self.map_object.create_completeness_sensor_map_fig(),
                    dash.no_update,
                )
