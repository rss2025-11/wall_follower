#!/usr/bin/env python3
"""
Example script demonstrating how to use the ROS2 bag analysis tools.
"""

import os
import numpy as np
from pathlib import Path

from analyze import BagReader
from visualization import (
    time_series_plot,
    histogram_plot,
    plot_laser_scan,
    plot_trajectory,
    plot_velocity_profile,
    save_plots,
    convert_timestamps_to_duration,
)


def main():
    """Main function to demonstrate the bag analysis tools."""
    bag_path = "rosbag_single_pd_2"
    topics_of_interest = [
        "/distance_from_wall",
        "/scan",
        "/vesc/high_level/input/nav_0",
    ]
    output_dir = bag_path + "_analysis_output"

    desired_distance = 0.5

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing bag file: {bag_path}")
    print(f"Output directory: {output_dir}")

    # Step 1: Extract data from the bag file
    reader = BagReader(bag_path)

    # Get all topics and types
    topic_types = reader.get_topics_and_types()
    print("\nAvailable topics:")
    for topic, msg_type in topic_types.items():
        print(f"  - {topic} ({msg_type})")

    # extract the relevant data
    data = reader.get_messages_by_topic(topics_of_interest)

    # Create some plots
    plots_dir = output_dir + "/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plots = []

    values, timestamps = zip(*data["/distance_from_wall"])
    values = [v.data for v in values]
    values = desired_distance - np.array(values)
    timestamps = convert_timestamps_to_duration(np.array(timestamps))
    dist_plot = time_series_plot(
        timestamps,
        values,
        "Error over time",
        "Meters",
        y_range=(-2.0, 2.0),
    )
    plots.append(dist_plot)

    values, timestamps = zip(*data["/vesc/high_level/input/nav_0"])
    values = [v.drive.steering_angle for v in values]
    values = np.array(values)
    timestamps = convert_timestamps_to_duration(np.array(timestamps))
    nav_plot = time_series_plot(
        timestamps,
        values,
        "Steering angle over time",
        "Radians",
        y_range=(-2.0, 2.0),
    )
    plots.append(nav_plot)

    save_plots(plots, plots_dir)

    print(f"Total error: {np.sum(np.abs(values))}")
    print(f"Variance of steering angle: {np.var(values)}")
    print(f"Mean of steering angle: {np.mean(values)}")
    
    # Calculate accuracy metric that decreases with error
    # For each timestep, accuracy is:
    # 100% if error is 0
    # Linear decrease down to 0% when error equals desired_distance
    errors = np.abs(values)
    accuracies = np.maximum(0, (1 - errors/desired_distance)) * 100
    mean_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {mean_accuracy:.2f}%")

    # Step 3: Print some statistics
    # print("\nBag statistics:")
    # duration = reader.get_bag_duration()
    # print(f"  - Duration: {duration:.2f} seconds")

    # message_counts = reader.get_message_count()
    # print("  - Message counts:")
    # for topic, count in message_counts.items():
    #     print(f"    - {topic}: {count} messages")

    print("\nAnalysis complete!")

    reader.close()


if __name__ == "__main__":
    main()
