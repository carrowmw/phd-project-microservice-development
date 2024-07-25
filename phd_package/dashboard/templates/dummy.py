# dashboard/utils/templates/dummy.py

import numpy as np
import pandas as pd
import dash
from dash import dcc
import dash_bootstrap_components as dbc

import plotly.express as px


class TabTemplateDummy:
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

    def dummy_graph(self):
        custom_colors = [
            "#051435",
            "#003a65",
            "#0073bc",
            "#29b7ea",
            "#8ed8f8",
            "#ffffff",
            "#da1a35",
            "#00a39b",
            "#fdc82f",
            "#92acca",
            "#aac7dc",
            "#c0d3e8",
            "#cee7f7",
            "#f1f9fe",
            "#f6c3cb",
            "#fae4ea",
            "#fef5f8",
            "#c3e4e8",
            "#e4f3f5",
            "#f5fafb",
            "#feedbe",
        ]
        # Create a DataFrame for plotting
        color_indices = np.array([[i] for i in range(len(custom_colors))])
        df = pd.DataFrame(color_indices, columns=["index"])
        df["colors"] = custom_colors

        # Use px.imshow to create the plot
        fig = px.imshow(
            df["index"].values.reshape(-1, 1),
            color_continuous_scale=custom_colors,
            labels=dict(color="Color Index"),
            title="Custom Color Palette",
        )

        # Adding annotations for each color
        for i, color in enumerate(custom_colors):
            fig.add_annotation(
                x=0,
                y=i,
                text=color,
                showarrow=False,
                yshift=0,
                font=dict(color="white" if i not in [5, 13, 16, 19] else "black"),
                xanchor="center",
                yanchor="middle",
            )

        fig.update_layout(
            coloraxis_showscale=False,
            title_text="Custom Color Palette",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
        )
        return fig

    def get_tab(self):
        return dbc.Tab(
            label=self.tab_label,
            tab_id=self.tab_id,
            children=[
                dcc.Graph(
                    id="dummy_graph",
                    figure=self.dummy_graph(),
                    className="graph-element",
                ),
            ],
            className="tab-element",
        )

    def setup_callbacks(self):
        pass  # No callbacks here anymore
