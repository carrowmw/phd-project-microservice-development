# dashboard/utils/dashboard_helper.py

from typing import Callable
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from utils.config_helper import get_horizon
from dashboard.data import CustomDashboardData
from dashboard.graphs import PipelineChartCreator
from dashboard.utils.color_helper import (
    base_colors,
    completeness_color_scale,
    freshness_color_scale,
    plot_color_gradient,
)


# class CardTemplate:
#     def __init__(
#         self,
#         container_id: str,
#         card_id: str,
#         card_label: str,
#         card_variable: str,
#         app: dash.Dash,
#     ):
#         self.container_id = container_id
#         self.card_id = card_id
#         self.card_label = card_label
#         self.card_variable = card_variable
#         self.app = app

#     def get_layout(self):
#         return dbc.Card(
#             id=self.card_id,
#             children=dbc.CardBody(
#                 [
#                     html.H4(
#                         children=self.card_label,
#                         className="card-title",
#                     ),
#                     html.P(
#                         id=self.card_variable,
#                         className="card-text",
#                     ),
#                 ],
#                 className="cardbody-element",
#             ),
#             className="card-element",
#         )

#     def get_container(self):
#         return dbc.Container(
#             id=self.container_id,
#             children=self.get_layout(),
#             className="container-element",
#         )

#     def setup_callbacks(self):
#         pass  # To be implemented in the derived classes


# class SensorNameCardTemplate(CardTemplate):
#     def __init__(
#         self,
#         container_id: str,
#         card_id: str,
#         card_label: str,
#         card_variable: str,
#         map_id: str,
#         app: dash.Dash,
#     ):
#         super().__init__(container_id, card_id, card_label, card_variable, app)
#         self.map_id = map_id
#         self.data = CustomDashboardData()
#         self.random_sensor = self.data.get_random_sensor()

#     def setup_callbacks(self):
#         @self.app.callback(
#             Output(self.card_variable, "children"),
#             Input(self.map_id, "clickData"),
#         )
#         def update_card(map_click_data):
#             print("SensorNameCardTemplate.setup_callbacks()")
#             if map_click_data is None:
#                 sensor_name = self.random_sensor
#             else:
#                 sensor_name = map_click_data["points"][0]["text"]
#             return f"{sensor_name}"


# class QualityMetricsCardTemplate(CardTemplate):
#     def __init__(
#         self,
#         container_id: str,
#         card_id: str,
#         card_label: str,
#         card_variable: str,
#         map_id: str,
#         dropdown_id: str,
#         app: dash.Dash,
#     ):
#         super().__init__(container_id, card_id, card_label, card_variable, app)
#         self.map_id = map_id
#         self.dropdown_id = dropdown_id
#         self.data = CustomDashboardData()
#         self.random_sensor = self.data.get_random_sensor()
#         self.pipeline_chart_creator = PipelineChartCreator()

#     def setup_callbacks(self):
#         @self.app.callback(
#             Output(self.card_variable, "children"),
#             [Input(self.map_id, "clickData"), Input(self.dropdown_id, "value")],
#         )
#         def update_card(map_click_data, selected_layer):
#             print("QualityMetricsCardTemplate.setup_callbacks()")
#             if map_click_data is None:
#                 sensor_name = self.random_sensor
#             else:
#                 sensor_name = map_click_data["points"][0]["text"]

#             if selected_layer == "completeness":
#                 metric_value = self.pipeline_chart_creator.create_completeness_metric(
#                     sensor_name
#                 )
#             elif selected_layer == "freshness":
#                 metric_value = self.pipeline_chart_creator.create_freshness_metric(
#                     sensor_name
#                 )
#             else:
#                 None

#             return metric_value


# class DropdownTemplate:
#     def __init__(
#         self,
#         dropdown_id: str,
#         map_id: str,
#         legend_id: str,
#         map_object: Callable[[str], dict],
#         app: dash.Dash,
#     ):
#         self.dropdown_id = dropdown_id
#         self.map_id = map_id
#         self.legend_id = legend_id
#         self.map_object = map_object
#         self.app = app
#         self.data = CustomDashboardData()

#     def get_layout(self):
#         return dcc.Dropdown(
#             id=self.dropdown_id,
#             options=[
#                 {"label": "Completeness", "value": "completeness"},
#                 {"label": "Freshness", "value": "freshness"},
#             ],
#             value="completeness",
#             className="dropdown-element",
#         )

#     def setup_callbacks(self):
#         @self.app.callback(
#             [Output(self.map_id, "figure"), Output(self.legend_id, "figure")],
#             Input(self.dropdown_id, "value"),
#         )
#         def update_map(selected_layer):
#             print(
#                 "DropdownTemplate.setup_callbacks() - Layer selected:", selected_layer
#             )
#             if selected_layer == "completeness":
#                 colorscale = completeness_color_scale()
#                 legend = plot_color_gradient(selected_layer, colorscale)
#                 map_fig = self.map_object.create_completeness_sensor_map_fig()
#                 return map_fig, legend
#             elif selected_layer == "freshness":
#                 tick_text = self.data.get_freshness_legend_tick_text()
#                 colorscale = freshness_color_scale()
#                 legend = plot_color_gradient(selected_layer, colorscale, tick_text)
#                 map_fig = self.map_object.create_freshness_sensor_map_fig()
#                 return map_fig, legend
#             else:
#                 return (
#                     self.map_object.create_completeness_sensor_map_fig(),
#                     dash.no_update,
#                 )


# class ContainerTemplateMapLegend:
#     def __init__(
#         self,
#         container_id: str,
#         legend_id: str,
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.container_id = container_id
#         self.card_label = "Map Legend"
#         self.legend_id = legend_id

#     def get_layout(self):
#         return dbc.Card(
#             id="legend_card",
#             children=dbc.CardBody(
#                 [
#                     html.H4(
#                         children=self.card_label,
#                         className="card-title",
#                     ),
#                     dcc.Graph(
#                         id=self.legend_id,
#                         className="legend-element",
#                     ),
#                 ],
#                 className="cardbody-element",
#             ),
#             className="card-element",
#         )

#     def get_container(self):
#         return dbc.Container(
#             id=self.container_id,
#             children=self.get_layout(),
#             className="container-element",
#         )

#     def setup_callbacks(self):
#         pass  # No callbacks here anymore


# class ContainerTemplateMap:
#     def __init__(
#         self,
#         container_id: str,
#         map_id: str,
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.container_id = container_id
#         self.map_id = map_id

#     def get_layout(self):
#         return dbc.Row(
#             id="map_row",
#             children=[
#                 dcc.Graph(
#                     id=self.map_id,
#                     className="graph-element map_graph",
#                 ),
#             ],
#             className="row-element",
#         )

#     def get_container(self):
#         return dbc.Container(
#             id=self.container_id,
#             children=self.get_layout(),
#             className="container-element",
#         )

#     def setup_callbacks(self):
#         pass  # No callbacks here anymore


# class TabTemplateDoubleGraph:
#     def __init__(
#         self,
#         tabs_id: str,
#         map_id: str,
#         tab_label: str,
#         tab_id: str,
#         graph_id_a: str,
#         graph_a_func: Callable[[str], go.Figure],
#         graph_id_b: str,
#         graph_b_func: Callable[[str], go.Figure],
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.tabs_id = tabs_id
#         self.map_id = map_id
#         self.tab_label = tab_label
#         self.tab_id = tab_id
#         self.graph_id_a = graph_id_a
#         self.graph_a = graph_a_func  # Store method reference
#         self.graph_id_b = graph_id_b
#         self.graph_b = graph_b_func  # Store method reference
#         self.data = CustomDashboardData()
#         self.random_sensor = self.data.get_random_sensor()

#     def get_layout(self):
#         return dbc.Row(
#             id="double_graph_row",
#             children=[
#                 dcc.Graph(
#                     id=self.graph_id_a,
#                     className="graph-element double_graph",
#                 ),
#                 dcc.Graph(
#                     id=self.graph_id_b,
#                     className="graph-element double_graph",
#                 ),
#             ],
#             className="row-element",
#         )

#     def get_tab(self):
#         return dbc.Tab(
#             id="tab",
#             label=self.tab_label,
#             tab_id=self.tab_id,
#             children=self.get_layout(),
#             className="tab-element",
#         )

#     def setup_callbacks(self):
#         @self.app.callback(
#             [Output(self.graph_id_a, "figure"), Output(self.graph_id_b, "figure")],
#             [
#                 Input(self.tabs_id, "active_tab"),
#                 Input(self.map_id, "clickData"),
#             ],
#         )
#         def update_tab(active_tab, map_click_data):
#             print("TabTemplateDoubleGraph.setup_callbacks()")
#             if map_click_data is None:
#                 sensor_name = self.random_sensor
#             else:
#                 sensor_name = map_click_data["points"][0]["text"]

#             if active_tab == self.tab_id:
#                 graph_a = self.graph_a(sensor_name)
#                 graph_b = self.graph_b(sensor_name)
#                 return graph_a, graph_b
#             else:
#                 return dash.no_update, dash.no_update


# class TabTemplateSingleTable:
#     def __init__(
#         self,
#         tabs_id: str,
#         map_id: str,
#         tab_label: str,
#         tab_id: str,
#         table_id: str,
#         table_func: Callable[[str], pd.DataFrame],
#         style_table: dict,
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.tabs_id = tabs_id
#         self.map_id = map_id
#         self.tab_label = tab_label
#         self.tab_id = tab_id
#         self.table_id = table_id
#         self.table_func = table_func  # Store method reference
#         self.style_table = style_table
#         self.data = CustomDashboardData()
#         self.random_sensor = self.data.get_random_sensor()

#     def get_layout(self):
#         return dbc.Row(
#             id="table_row",
#             children=[
#                 dash_table.DataTable(
#                     id=self.table_id,
#                     columns=[
#                         {
#                             "name": i,
#                             "id": i,
#                             "type": "numeric",
#                             "format": {"specifier": ".2f"},
#                         }
#                         for i in self.table_func(self.random_sensor).columns
#                         if i != "index"
#                     ],
#                     style_table=self.style_table,
#                     data=self.table_func(self.random_sensor).to_dict("records"),
#                 )
#             ],
#             className="row-element",
#         )

#     def get_tab(self):
#         return dbc.Tab(
#             label=self.tab_label,
#             tab_id=self.tab_id,
#             children=self.get_layout(),
#             className="tab-element",
#         )

#     def setup_callbacks(self):
#         @self.app.callback(
#             Output(self.table_id, "data"),
#             [Input(self.tabs_id, "active_tab"), Input(self.map_id, "clickData")],
#         )
#         def update_tab(active_tab, map_click_data):
#             print("TabTemplateSingleTable.setup_callbacks()")
#             if map_click_data is None:
#                 sensor_name = self.random_sensor
#             else:
#                 sensor_name = map_click_data["points"][0]["text"]

#             if active_tab == self.tab_id:
#                 data = self.table_func(sensor_name)
#                 return data.to_dict("records")
#             else:
#                 return dash.no_update


# class TabTemplateDataLoaderViewer:
#     def __init__(
#         self,
#         tabs_id: str,
#         map_id: str,
#         tab_label: str,
#         tab_id: str,
#         graph_id: str,
#         graph_func: Callable[[str], go.Figure],
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.tabs_id = tabs_id
#         self.map_id = map_id
#         self.tab_label = tab_label
#         self.tab_id = tab_id
#         self.graph_id = graph_id
#         self.graph = graph_func  # Store method reference
#         self.data = CustomDashboardData()
#         self.random_sensor = self.data.get_random_sensor()

#     def get_layout(self):
#         return dbc.Row(
#             id="graph_row",
#             children=[
#                 dcc.Store(id="prev-sensor-name"),
#                 dcc.Graph(
#                     id=self.graph_id,
#                     className="graph-element",
#                 ),
#                 html.Div(
#                     children=[
#                         dbc.Button(
#                             id="prev-window-button",
#                             children="Previous",
#                             n_clicks=0,
#                             color=base_colors["Mid Blue"],
#                             className="button-element",
#                         ),
#                         html.Span(
#                             id="frame-counter",
#                             children="Sequence 1 out of 1",
#                             className="frame-counter-element",
#                             style={"verticalAlign": "middle"},
#                         ),
#                         dbc.Button(
#                             id="next-window-button",
#                             children="Next",
#                             n_clicks=0,
#                             color=base_colors["Mid Blue"],
#                             className="button-element",
#                         ),
#                     ],
#                     className="button-frame-container",  # Apply Flexbox here
#                 ),
#             ],
#             className="row-element",
#         )

#     def get_tab(self):
#         return dbc.Tab(
#             label=self.tab_label,
#             tab_id=self.tab_id,
#             children=self.get_layout(),
#             className="tab-element",
#         )

#     def setup_callbacks(self):
#         @self.app.callback(
#             [
#                 Output(self.graph_id, "figure"),
#                 Output("frame-counter", "children"),
#                 Output(
#                     "prev-sensor-name", "data"
#                 ),  # This updates the dcc.Store component
#             ],
#             [
#                 Input(self.map_id, "clickData"),
#                 Input(self.tabs_id, "active_tab"),
#                 Input("prev-window-button", "n_clicks"),
#                 Input("next-window-button", "n_clicks"),
#             ],
#             [
#                 State("frame-counter", "children"),  # Store the current frame counter
#                 State("prev-sensor-name", "data"),  # Store the previous sensor name
#             ],
#         )
#         def update_tab(
#             map_click_data,
#             active_tab,
#             prev_clicks,
#             next_clicks,
#             frame_counter,
#             prev_sensor_name,
#         ):
#             print("TabTemplateDataLoaderViewer.setup_callbacks()")
#             if map_click_data is None:
#                 sensor_name = self.random_sensor
#             else:
#                 sensor_name = map_click_data["points"][0]["text"]

#             if active_tab == self.tab_id:
#                 # Get the number of frames from the data
#                 data = self.data.get_training_windows()
#                 sensor_index = data[3].index(sensor_name)
#                 features = data[0][sensor_index]
#                 labels = data[1][sensor_index]
#                 number_of_items = len(features)

#                 if sensor_name != prev_sensor_name:
#                     current_frame = 1
#                     number_of_clicks = 0
#                 else:

#                     current_frame = int(frame_counter.split(" ")[1])
#                     callback_context = dash.ctx
#                     if not callback_context.triggered:
#                         # current_frame = 1
#                         number_of_clicks = 0
#                     else:
#                         button_id = callback_context.triggered[0]["prop_id"].split(".")[
#                             0
#                         ]
#                         if button_id == "next-window-button":
#                             number_of_clicks = 1
#                         elif button_id == "prev-window-button":
#                             number_of_clicks = -1
#                         else:
#                             number_of_clicks = 0

#                 current_index = (current_frame + number_of_clicks - 1) % number_of_items
#                 if current_index == -1:
#                     current_index = number_of_items - 1
#                 new_frame_counter = (
#                     f"Sequence {current_index + 1} out of {number_of_items}"
#                 )

#                 # Create the data for the current frame
#                 x_values = list(range(len(features[current_index])))
#                 print("x_values: ", x_values)
#                 y_values = list(features[current_index])
#                 label_value = labels[current_index]
#                 horizon = get_horizon()

#                 # Create the graph for the current frame
#                 fig = {
#                     "data": [
#                         go.Scatter(
#                             x=x_values, y=y_values, mode="lines", name="Training Window"
#                         ),
#                         go.Scatter(
#                             x=[len(x_values) + horizon],
#                             y=label_value,
#                             mode="markers",
#                             name="Label",
#                         ),
#                     ],
#                     "layout": {
#                         "xaxis": {"title": "X-axis"},
#                         "yaxis": {"title": "Y-axis", "range": [0, 1]},
#                     },
#                 }
#                 # graph = self.graph(sensor_name, frame_counter)
#                 return fig, new_frame_counter, sensor_name
#             else:
#                 return dash.no_update, dash.no_update, dash.no_update


# class TabTemplateDummy:
#     def __init__(
#         self,
#         tabs_id: str,
#         tab_label: str,
#         tab_id: str,
#         app: dash.Dash,
#     ):
#         self.app = app
#         self.tabs_id = tabs_id
#         self.tab_label = tab_label
#         self.tab_id = tab_id

#     def dummy_graph(self):
#         custom_colors = [
#             "#051435",
#             "#003a65",
#             "#0073bc",
#             "#29b7ea",
#             "#8ed8f8",
#             "#ffffff",
#             "#da1a35",
#             "#00a39b",
#             "#fdc82f",
#             "#92acca",
#             "#aac7dc",
#             "#c0d3e8",
#             "#cee7f7",
#             "#f1f9fe",
#             "#f6c3cb",
#             "#fae4ea",
#             "#fef5f8",
#             "#c3e4e8",
#             "#e4f3f5",
#             "#f5fafb",
#             "#feedbe",
#         ]
#         # Create a DataFrame for plotting
#         color_indices = np.array([[i] for i in range(len(custom_colors))])
#         df = pd.DataFrame(color_indices, columns=["index"])
#         df["colors"] = custom_colors

#         # Use px.imshow to create the plot
#         fig = px.imshow(
#             df["index"].values.reshape(-1, 1),
#             color_continuous_scale=custom_colors,
#             labels=dict(color="Color Index"),
#             title="Custom Color Palette",
#         )

#         # Adding annotations for each color
#         for i, color in enumerate(custom_colors):
#             fig.add_annotation(
#                 x=0,
#                 y=i,
#                 text=color,
#                 showarrow=False,
#                 yshift=0,
#                 font=dict(color="white" if i not in [5, 13, 16, 19] else "black"),
#                 xanchor="center",
#                 yanchor="middle",
#             )

#         fig.update_layout(
#             coloraxis_showscale=False,
#             title_text="Custom Color Palette",
#             xaxis_showticklabels=False,
#             yaxis_showticklabels=False,
#             xaxis_zeroline=False,
#             yaxis_zeroline=False,
#         )
#         return fig

#     def get_tab(self):
#         return dbc.Tab(
#             label=self.tab_label,
#             tab_id=self.tab_id,
#             children=[
#                 dcc.Graph(
#                     id="dummy_graph",
#                     figure=self.dummy_graph(),
#                     className="graph-element",
#                 ),
#             ],
#             className="tab-element",
#         )

#     def setup_callbacks(self):
#         pass  # No callbacks here anymore
