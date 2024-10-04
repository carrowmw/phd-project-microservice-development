# dashboard/utils/templates/tables.py

from typing import Callable
import pandas as pd
import dash
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


from ..data import CustomDashboardData


class TabTemplateSingleTable:
    def __init__(
        self,
        tabs_id: str,
        map_id: str,
        tab_label: str,
        tab_id: str,
        table_id: str,
        table_func: Callable[[str], pd.DataFrame],
        style_table: dict,
        app: dash.Dash,
    ):
        self.app = app
        self.tabs_id = tabs_id
        self.map_id = map_id
        self.tab_label = tab_label
        self.tab_id = tab_id
        self.table_id = table_id
        self.table_func = table_func  # Store method reference
        self.style_table = style_table
        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()

    def get_layout(self):
        return dbc.Row(
            id="table_row",
            children=[
                dash_table.DataTable(
                    id=self.table_id,
                    columns=[
                        {
                            "name": i,
                            "id": i,
                            "type": "numeric",
                            "format": {"specifier": ".2f"},
                        }
                        for i in self.table_func(self.random_sensor).columns
                        if i != "index"
                    ],
                    style_table=self.style_table,
                    data=self.table_func(self.random_sensor).to_dict("records"),
                )
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
            Output(self.table_id, "data"),
            [Input(self.tabs_id, "active_tab"), Input(self.map_id, "clickData")],
        )
        def update_tab(active_tab, map_click_data):
            # print("TabTemplateSingleTable.setup_callbacks()")
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"].split("<br>")[0]

            if active_tab == self.tab_id:
                data = self.table_func(sensor_name)
                return data.to_dict("records")
            else:
                return dash.no_update
