#!/usr/bin/env python3
"""
Visualization Functions for ROS2 Bag Analysis

This module contains functions for creating visualizations from ROS2 bag data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Local imports
from analyze import BagReader

# Set Seaborn style
sns.set_theme(style="whitegrid")


def setup_plot_style():
    """Set up the default plot style."""
    # Set the figure size
    plt.rcParams["figure.figsize"] = (12, 6)

    # Set the font size
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    # Set the color palette
    sns.set_palette("muted")


def time_series_plot(
    timestamps: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    xlabel: str = "Time (s)",
    y_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create a time series plot.

    Args:
        timestamps: Array of timestamps
        values: Array of values
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        y_range: Y-axis range (min, max)
    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots()

    # Handle multi-dimensional data
    if values.ndim > 1 and values.shape[1] <= 10:
        # Plot each dimension as a separate line
        for i in range(values.shape[1]):
            ax.plot(timestamps, values[:, i], label=f"Dimension {i}")
        ax.legend()
    else:
        # Plot single dimension
        ax.plot(timestamps, values)

    if y_range:
        ax.set_ylim(y_range)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


def multi_time_series_plot(
    data_dict: Dict[str, Dict[str, np.ndarray]],
    title: str,
    ylabel: str,
    xlabel: str = "Time (s)",
) -> plt.Figure:
    """
    Create a plot with multiple time series.

    Args:
        data_dict: Dictionary mapping series names to dictionaries with 'timestamps' and 'values'
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots()

    for name, data in data_dict.items():
        ax.plot(data["timestamps"], data["values"], label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


def histogram_plot(
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 30,
    kde: bool = True,
) -> plt.Figure:
    """
    Create a histogram plot.

    Args:
        values: Array of values
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins
        kde: Whether to show kernel density estimate

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots()

    # Create histogram with Seaborn
    sns.histplot(values, bins=bins, kde=kde, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


def scatter_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    hue: Optional[np.ndarray] = None,
    hue_label: Optional[str] = None,
) -> plt.Figure:
    """
    Create a scatter plot.

    Args:
        x_values: Array of x values
        y_values: Array of y values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        hue: Array of values to use for point colors
        hue_label: Label for the color bar

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots()

    # Create scatter plot with Seaborn
    if hue is not None:
        scatter = sns.scatterplot(
            x=x_values, y=y_values, hue=hue, palette="viridis", ax=ax
        )

        # Add color bar if hue is provided
        if hue_label:
            norm = plt.Normalize(hue.min(), hue.max())
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(hue_label)
    else:
        scatter = sns.scatterplot(x=x_values, y=y_values, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


def heatmap_plot(
    data: np.ndarray,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    cmap: str = "viridis",
    annot: bool = True,
) -> plt.Figure:
    """
    Create a heatmap plot.

    Args:
        data: 2D array of values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        x_labels: Labels for x-axis ticks
        y_labels: Labels for y-axis ticks
        cmap: Colormap name
        annot: Whether to annotate cells with values

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with Seaborn
    sns.heatmap(
        data, annot=annot, cmap=cmap, ax=ax, xticklabels=x_labels, yticklabels=y_labels
    )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Tight layout
    fig.tight_layout()

    return fig


def polar_plot(
    angles: np.ndarray,
    radii: np.ndarray,
    title: str,
) -> plt.Figure:
    """
    Create a polar plot (useful for visualizing LiDAR data).

    Args:
        angles: Array of angles in radians
        radii: Array of radii
        title: Plot title

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    # Create polar plot
    ax.scatter(angles, radii, s=2, alpha=0.5)

    ax.set_title(title)
    ax.grid(True)

    # Tight layout
    fig.tight_layout()

    return fig


def plot_laser_scan(
    bag_reader: BagReader,
    topic: str,
    index: int = -1,
    title: str = "LiDAR Scan",
) -> plt.Figure:
    """
    Create a polar plot for LiDAR data.

    Args:
        bag_reader: BagReader instance
        topic: Topic name for LaserScan messages
        index: Index of the scan to plot (-1 for the last scan)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Get all messages for this topic
    messages = []
    for _, msg, _ in bag_reader.read_messages([topic]):
        messages.append(msg)

    if not messages:
        raise ValueError(f"No messages found for topic: {topic}")

    # Get the scan message
    if index < 0:
        index = len(messages) + index

    if index < 0 or index >= len(messages):
        raise IndexError(
            f"Index {index} out of range for topic {topic} with {len(messages)} messages"
        )

    msg = messages[index]

    # Extract scan parameters
    angle_min = msg.angle_min
    angle_max = msg.angle_max
    angle_increment = msg.angle_increment
    ranges = np.array(msg.ranges)

    # Calculate angles
    angles = np.arange(angle_min, angle_max + angle_increment, angle_increment)

    # Ensure angles and ranges have the same length
    angles = angles[: len(ranges)]

    # Create polar plot
    return polar_plot(angles, ranges, title)


def plot_trajectory(
    bag_reader: BagReader,
    topic: str,
    position_x_field: str = "pose.pose.position.x",
    position_y_field: str = "pose.pose.position.y",
    title: str = "Robot Trajectory",
    color_by_time: bool = True,
) -> plt.Figure:
    """
    Create a plot of the robot's trajectory.

    Args:
        bag_reader: BagReader instance
        topic: Topic name for Odometry messages
        position_x_field: Field path for x position
        position_y_field: Field path for y position
        title: Plot title
        color_by_time: Whether to color the trajectory by time

    Returns:
        Matplotlib figure
    """
    # Get position data
    x_values = bag_reader.get_message_field_values(topic, position_x_field)
    y_values = bag_reader.get_message_field_values(topic, position_y_field)

    if not x_values or not y_values:
        raise ValueError(f"No position data found for topic: {topic}")

    # Extract values and timestamps
    x_positions = np.array([x for x, _ in x_values])
    y_positions = np.array([y for y, _ in y_values])
    timestamps = np.array([t for _, t in x_values])

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create trajectory plot
    if color_by_time:
        # Color by time
        points = ax.scatter(
            x_positions, y_positions, c=timestamps, cmap="viridis", s=10, alpha=0.7
        )
        plt.colorbar(points, ax=ax, label="Time (s)")
    else:
        # Plot as a line
        ax.plot(x_positions, y_positions, "-o", markersize=2, alpha=0.7)

    # Mark start and end points
    ax.plot(x_positions[0], y_positions[0], "go", markersize=10, label="Start")
    ax.plot(x_positions[-1], y_positions[-1], "ro", markersize=10, label="End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Tight layout
    fig.tight_layout()

    return fig


def plot_velocity_profile(
    bag_reader: BagReader,
    topic: str,
    linear_vel_field: str,
    angular_vel_field: str,
    title: str = "Velocity Profile",
) -> plt.Figure:
    """
    Create a plot of the robot's velocity profile.

    Args:
        bag_reader: BagReader instance
        topic: Topic name for velocity messages
        linear_vel_field: Field path for linear velocity
        angular_vel_field: Field path for angular velocity
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Get velocity data
    linear_values = bag_reader.get_message_field_values(topic, linear_vel_field)
    angular_values = bag_reader.get_message_field_values(topic, angular_vel_field)

    if not linear_values or not angular_values:
        raise ValueError(f"No velocity data found for topic: {topic}")

    # Extract values and timestamps
    linear_velocities = np.array([v for v, _ in linear_values])
    angular_velocities = np.array([v for v, _ in angular_values])
    timestamps = np.array([t for _, t in linear_values])

    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot linear velocity
    ax1.plot(timestamps, linear_velocities, "b-", label="Linear Velocity")
    ax1.set_ylabel("Linear Velocity (m/s)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot angular velocity
    ax2.plot(timestamps, angular_velocities, "r-", label="Angular Velocity")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Set title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

    return fig


def plot_field_time_series(
    bag_reader: BagReader,
    topic: str,
    field_path: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> plt.Figure:
    """
    Create a time series plot for a specific field.

    Args:
        bag_reader: BagReader instance
        topic: Topic name
        field_path: Field path
        title: Plot title (if None, use topic and field)
        ylabel: Y-axis label (if None, use field)

    Returns:
        Matplotlib figure
    """
    # Get field values
    values = bag_reader.get_message_field_values(topic, field_path)

    if not values:
        raise ValueError(f"No data found for field {field_path} in topic {topic}")

    # Extract values and timestamps
    field_values = np.array([v for v, _ in values])
    timestamps = np.array([t for _, t in values])

    # Set default title and ylabel if not provided
    if title is None:
        title = f"{topic} - {field_path}"
    if ylabel is None:
        ylabel = field_path

    # Create time series plot
    return time_series_plot(
        timestamps=timestamps,
        values=field_values,
        title=title,
        ylabel=ylabel,
    )


def save_plots(plots: List[plt.Figure], folder_path: Path):
    os.makedirs(folder_path, exist_ok=True)
    for i, plot in enumerate(plots):
        plot.savefig(folder_path + f"/{i}.png", dpi=300, bbox_inches="tight")


def convert_timestamps_to_duration(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert timestamps to duration, starting from 0.
    """
    return timestamps - timestamps[0]
