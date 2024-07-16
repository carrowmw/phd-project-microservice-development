# dashboard/utils/templates/map.py

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc


class ContainerTemplateMapLegend:
    def __init__(
        self,
        container_id: str,
        legend_id: str,
        app: dash.Dash,
    ):
        self.app = app
        self.container_id = container_id
        self.card_label = "Map Legend"
        self.legend_id = legend_id

    def get_layout(self):
        return dbc.Card(
            id="legend_card",
            children=dbc.CardBody(
                [
                    html.H4(
                        children=self.card_label,
                        className="card-title",
                    ),
                    dcc.Graph(
                        id=self.legend_id,
                        className="legend-element",
                    ),
                ],
                className="cardbody-element",
            ),
            className="card-element",
        )

    def get_container(self):
        return dbc.Container(
            id=self.container_id,
            children=self.get_layout(),
            className="container-element",
        )

    def setup_callbacks(self):
        pass  # No callbacks here anymore


class ContainerTemplateMap:
    def __init__(
        self,
        container_id: str,
        map_id: str,
        app: dash.Dash,
    ):
        self.app = app
        self.container_id = container_id
        self.map_id = map_id

    def get_layout(self):
        return dbc.Row(
            id="map_row",
            children=[
                dcc.Graph(
                    id=self.map_id,
                    className="graph-element map_graph",
                ),
            ],
            className="row-element",
        )

    def get_container(self):
        return dbc.Container(
            id=self.container_id,
            children=self.get_layout(),
            className="container-element",
        )

    def setup_callbacks(self):
        pass  # No callbacks here anymore
