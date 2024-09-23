# dashboard/utils/templates/dummy.py

import numpy as np
import pandas as pd
import dash
from dash import dcc
import dash_bootstrap_components as dbc

import plotly.express as px


class TabTemplateMetaData:
    def __init__(
        self,
        tabs_id: str,
        tab_label: str,
        tab_id: str,
        app: dash.Dash,
    ):
        self.app = app
        self.tabs_id = tabs_id
        self.tab_label = tab_label
        self.tab_id = tab_id

    def metadata_table(self):
        # hyperparameter table logic goes here
        return None

    def get_tab(self):
        return dbc.Tab(
            label=self.tab_label,
            tab_id=self.tab_id,
            children=[
                dcc.Graph(
                    id="metadata_table",
                    figure=self.metadata_table(),
                    className="graph-element",
                ),
            ],
            className="tab-element",
        )

    def setup_callbacks(self):
        pass  # No callbacks here anymore
