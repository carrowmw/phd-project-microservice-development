# dashboard/utils/templates/dummy.py
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from ..data import CustomDashboardData
from dash import dash_table


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
        self.data = CustomDashboardData()
        self.metadata = self.data.get_metadata()

    def get_layout(self):
        metadata_df = pd.DataFrame(
            list(self.metadata.items()), columns=["Metric", "Value"]
        )
        return dbc.Row(
            id="metadata_row",
            children=[
                dash_table.DataTable(
                    data=metadata_df.to_dict("records"),
                    columns=[{"name": col, "id": col} for col in metadata_df.columns],
                    id="metadata_table",
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
        pass  # No callbacks here anymore
