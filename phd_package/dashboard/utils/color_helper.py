# dashboard/utils/color_helper.py

"""
This module contains helper functions for creating color scales and gradients.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import plotly.graph_objects as go


def completeness_color_scale():
    """
    Returns a color scale for completeness values.
    """
    color_scale = [
        [0, "#da1a35"],  # red
        [0.5, "#fdc82f"],  # yellow
        [1, "#00857e"],  # green
    ]
    return color_scale


def freshness_color_scale():
    """
    Returns a color scale for freshness values.
    """
    color_scale = [
        [0, "#0073bc"],  # blue
        [1, "#da1a35"],  # red
    ]
    return color_scale


base_colors = {
    "Dark Blue": "#051435",
    "Rich Blue": "#003a65",
    "Mid Blue": "#0073bc",
    "Bright Blue": "#29b7ea",
    "Light Blue": "#8ed8f8",
}

accent_colors = {
    "Red": "#da1a35",
    "Teal": "#00a39b",
    "Yellow": "#fdc82f",
}

category_colors = {
    "Mid Blue": "#0073bc",
    "Red": "#da1a35",
    "Yellow": "#fdc82f",
    "Light Blue": "#8ed8f8",
    "Teal": "#00a39b",
    "Dark Blue": "#051435",
}

blue_tints = {
    "Blue Tint 1": "#92acca",
    "Blue Tint 2": "#aac7dc",
    "Blue Tint 3": "#c0d3e8",
    "Blue Tint 4": "#cee7f7",
    "Blue Tint 5": "#f1f9fe",
}

red_tints = {
    "Red Tint 1": "#f6c3cb",
    "Red Tint 2": "#fae4ea",
    "Red Tint 3": "#fef5f8",
}

teal_tints = {
    "Teal Tint 1": "#c3e4e8",
    "Teal Tint 2": "#e4f3f5",
    "Teal Tint 3": "#f5fafb",
}

yellow_tints = {
    "Yellow Tint 1": "#feedbe",
}


def plot_color_gradient(name, colorscale):
    """
    Plots a color gradient for a given color scale.

    Args:
        name (str): The name of the color scale.
        colorscale (list): A list of colors to blend.
        legend_tick_text (str): The text to display on the legend ticks.

    Returns:
        go.Figure: A Plotly figure of the color gradient.
    """
    # Normalize the z values
    z = np.array([np.linspace(0, 1, 256)]).T
    y = np.linspace(0, 1, 256)
    tickvals = np.linspace(0, 1, 11)
    ticktext = [str(np.round(val, 2)) for val in tickvals]

    if name == "completeness":
        tickvals = np.linspace(0, 1, 11)
        ticktext = [str(np.round(val, 1) * 100) for val in tickvals]
        name = "Completeness (%)"

    if name == "freshness":
        tickvals = np.linspace(0, 1, 2)
        ticktext = [0, 1]
        name = "Freshness (Normalised Logarithmic Scale)"

    # Create a gradient heatmap
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                colorscale=colorscale,
                showscale=False,  # Hide the colorbar
                x=[0],  # Single column
                y=y,  # Corresponding to the scale values
                hoverinfo="text",  # Show the y-axis values on hover
            )
        ]
    )

    fig.update_layout(
        xaxis_showgrid=False,
        # width=10,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        xaxis_visible=False,
        yaxis_visible=True,  # Show the y-axis with scale values
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        # title=dict(text=name, font=dict(size=12)),
        margin=dict(t=40, b=40),
    )

    # Add vertical title on the left side
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-1.2,  # Adjust this value to position the title further left or right
        y=0.5,  # Middle of the y-axis
        text=name,
        showarrow=False,
        textangle=-90,  # Rotate the text to be vertical
        font=dict(size=14),
    )

    return fig
