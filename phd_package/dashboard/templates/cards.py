# dashboard/utils/templates/cards.py

import dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


from dashboard.data import CustomDashboardData
from dashboard.graphs import PipelineChartCreator


class CardTemplate:
    def __init__(
        self,
        container_id: str,
        card_id: str,
        card_label: str,
        card_variable: str,
        app: dash.Dash,
    ):
        self.container_id = container_id
        self.card_id = card_id
        self.card_label = card_label
        self.card_variable = card_variable
        self.app = app

    def get_layout(self):
        return dbc.Card(
            id=self.card_id,
            children=dbc.CardBody(
                [
                    html.H4(
                        children=self.card_label,
                        className="card-title",
                    ),
                    html.P(
                        id=self.card_variable,
                        className="card-text",
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
        pass  # To be implemented in the derived classes


class SensorNameCardTemplate(CardTemplate):
    def __init__(
        self,
        container_id: str,
        card_id: str,
        card_label: str,
        card_variable: str,
        map_id: str,
        app: dash.Dash,
    ):
        super().__init__(container_id, card_id, card_label, card_variable, app)
        self.map_id = map_id
        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()

    def setup_callbacks(self):
        @self.app.callback(
            Output(self.card_variable, "children"),
            Input(self.map_id, "clickData"),
        )
        def update_card(map_click_data):
            print("SensorNameCardTemplate.setup_callbacks()")
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"]
            return f"{sensor_name}"


class QualityMetricsCardTemplate(CardTemplate):
    def __init__(
        self,
        container_id: str,
        card_id: str,
        card_label: str,
        card_variable: str,
        map_id: str,
        dropdown_id: str,
        app: dash.Dash,
    ):
        super().__init__(container_id, card_id, card_label, card_variable, app)
        self.map_id = map_id
        self.dropdown_id = dropdown_id
        self.data = CustomDashboardData()
        self.random_sensor = self.data.get_random_sensor()
        self.pipeline_chart_creator = PipelineChartCreator()

    def get_layout(self):
        return dbc.Card(
            id=self.card_id,
            children=dbc.CardBody(
                [
                    html.H4(
                        children=self.card_label,
                        className="card-title quality-cardtitle-font",  # Add the custom class
                    ),
                    html.P(
                        id=self.card_variable,
                        className="card-text quality-cardbody-font",  # Add the custom class
                    ),
                ],
                className="cardbody-element",
            ),
            className="card-element",
        )

    def setup_callbacks(self):
        @self.app.callback(
            Output(self.card_variable, "children"),
            [Input(self.map_id, "clickData"), Input(self.dropdown_id, "value")],
        )
        def update_card(map_click_data, selected_layer):
            print("QualityMetricsCardTemplate.setup_callbacks()")
            if map_click_data is None:
                sensor_name = self.random_sensor
            else:
                sensor_name = map_click_data["points"][0]["text"]

            if selected_layer == "completeness":
                metric_value = self.pipeline_chart_creator.create_completeness_metric(
                    sensor_name
                )
            elif selected_layer == "freshness":
                metric_value = self.pipeline_chart_creator.create_freshness_metric(
                    sensor_name
                )
            else:
                None

            return metric_value
