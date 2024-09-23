import dash
import dash_bootstrap_components as dbc
from dash import html

from phd_package.pipeline import Pipeline

from .templates.tables import TabTemplateSingleTable
from .templates.dataloader_viewer import TabTemplateDataLoaderViewer
from .templates.dummy import TabTemplateDummy
from .templates.cards import (
    SensorNameCardTemplate,
    QualityMetricsCardTemplate,
)
from .templates.map import ContainerTemplateMap, ContainerTemplateMapLegend
from .templates.dropdown import DropdownTemplate
from .templates.double_graph import TabTemplateDoubleGraph
from .templates.metadata import TabTemplateMetaData
from .data import CustomDashboardData
from .graphs import PipelineChartCreator, SensorMapCreator


class SensorDashboardApp:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.map_creator = SensorMapCreator()

        self.map_container = ContainerTemplateMap(
            "map_container",
            "sensor_map",
            self.app,
        )
        self.legend_container = ContainerTemplateMapLegend(
            "legend_container",
            "sensor_map_legend",
            self.app,
        )

        self.dropdown = DropdownTemplate(
            "layer_dropdown",
            self.map_container.map_id,
            self.legend_container.legend_id,
            self.map_creator,
            self.app,
        )

        self.sensor_name_card = SensorNameCardTemplate(
            "sensor_name_container",
            "sensor_name_card",
            "Sensor Name",
            "selected-sensor-name",
            self.map_container.map_id,
            self.app,
        )

        self.quality_metrics_card = QualityMetricsCardTemplate(
            "quality_metrics_container",
            "quality_metrics_card",
            "Quality Metrics",
            "data-metric-value",
            self.map_container.map_id,
            self.dropdown.dropdown_id,
            self.app,
        )

        self.data_quality_tab = TabTemplateDoubleGraph(
            "tabs",
            "sensor_map",
            "Data Quality",
            "data_quality_tab",
            "data_quality_graph_a",
            PipelineChartCreator().create_latest_data_graph,
            "data_quality_graph_b",
            PipelineChartCreator().create_records_per_day_graph,
            self.app,
        )

        self.engineering_tab = TabTemplateSingleTable(
            "tabs",
            "sensor_map",
            "Engineering",
            "engineering_tab",
            "engineering_table",
            CustomDashboardData().get_engineering_table,
            {
                "height": "60vh",
                "width": "45vw",
                "margin": "50px auto",
                "overflowY": "auto",
            },
            self.app,
        )
        self.preprocessing_tab = TabTemplateSingleTable(
            "tabs",
            "sensor_map",
            "Preprocessing",
            "preprocessing_tab",
            "preprocessing_table",
            CustomDashboardData().get_preprocessing_table,
            {
                "height": "50vh",
                "width": "25vw",
                "margin": "50px auto",
                "overflowY": "auto",
            },
            self.app,
        )
        self.dataloader_tab = TabTemplateDataLoaderViewer(
            "tabs",
            "sensor_map",
            "Data Loader",
            "dataloader_tab",
            "dataloader_viewer",
            "dataloader_viewer",
            self.app,
        )
        self.model_performance_tab = TabTemplateDoubleGraph(
            "tabs",
            "sensor_map",
            "Model Performance",
            "model_performance_tab",
            "model_performance_graph_a",
            PipelineChartCreator().create_test_predictions_graph,
            "model_performance_graph_b",
            PipelineChartCreator().create_train_metrics_graph,
            self.app,
        )
        self.metadata_tab = TabTemplateMetaData(
            "tabs",
            "Metadata",
            "metadata_tab",
            self.app,
        )

        self.dummy_tab = TabTemplateDummy(
            "tabs",
            "Dummy",
            "dummy_tab",
            self.app,
        )

        print("Dashboard App Initialized")
        print("Legend id: ", self.legend_container.legend_id)
        print("Map id: ", self.map_container.map_id)

    def setup_layout(self):
        title_row = dbc.Row(
            id="title_row",
            children=[
                html.H2(
                    "Sensor Dashboard",
                    className="title-element",
                ),
                html.Img(
                    src="assets/white_logos/cdt_long.png",
                    className="logo-element cdt",
                ),
                html.Img(
                    src="assets/white_logos/newcastle.png",
                    className="logo-element newcastle",
                ),
            ],
            align="center",
            className="row-element",
        )

        left_content_column = dbc.Col(
            id="left_content_column",
            children=[
                self.dropdown.get_layout(),
                self.map_container.get_container(),
            ],
            width=5,
            className="column-element",
        )

        middle_content_column = dbc.Col(
            id="middle_content_column",
            children=[
                self.sensor_name_card.get_container(),
                self.quality_metrics_card.get_container(),
                self.legend_container.get_container(),
            ],
            width=2,
            className="column-element",
        )

        right_content_column = dbc.Col(
            id="right_content_column",
            children=[
                dbc.Tabs(
                    id="tabs",
                    children=[
                        self.data_quality_tab.get_tab(),
                        self.preprocessing_tab.get_tab(),
                        self.engineering_tab.get_tab(),
                        self.dataloader_tab.get_tab(),
                        self.model_performance_tab.get_tab(),
                        self.metadata_tab.get_tab(),
                        self.dummy_tab.get_tab(),
                    ],
                    className="tabs-element",
                ),
            ],
            width=5,
            className="column-element",
        )

        content_row = dbc.Row(
            id="content_row",
            children=[middle_content_column, left_content_column, right_content_column],
            align="center",
            className="row-element",
        )

        footer_row = dbc.Row(
            id="footer_row",
            children=[
                html.P(
                    "© 2024 Carrow Morris-Wiltshire. All rights reserved.",
                    className="footer-element",
                ),
            ],
            align="center",
            className="row-element",
        )

        self.app.layout = html.Div(
            id="main_container",
            children=[
                html.Div(children=[title_row, content_row, footer_row]),
            ],
        )

    def setup_callbacks(self):
        self.dropdown.setup_callbacks()
        self.sensor_name_card.setup_callbacks()
        self.quality_metrics_card.setup_callbacks()
        self.data_quality_tab.setup_callbacks()
        self.preprocessing_tab.setup_callbacks()
        self.engineering_tab.setup_callbacks()
        self.dataloader_tab.setup_callbacks()
        self.model_performance_tab.setup_callbacks()


if __name__ == "__main__":
    dashboard_app = SensorDashboardApp()
    dashboard_app.setup_layout()
    dashboard_app.setup_callbacks()
    dashboard_app.app.run_server(debug=True)
