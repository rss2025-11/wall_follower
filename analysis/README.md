# ROS2 Bag Analysis Tool

A tool for analyzing and visualizing data from ROS2 bag files.

## Overview

This project provides a set of tools for:

1. Reading and analyzing ROS2 bag files
2. Accessing fields in ROS2 messages
3. Creating visualizations of the data
4. Computing statistics and aggregating data

It's designed to be flexible and extensible, allowing you to analyze data from autonomous RC cars and other robotic systems.

## Features

- Read messages from specified topics in ROS2 bag files
- Access fields in ROS2 messages using dot notation (e.g., "pose.pose.position.x")
- Create various visualizations:
  - Time series plots
  - Histograms
  - Scatter plots
  - Heatmaps
  - Polar plots (for LiDAR data)
  - Trajectory plots
  - Velocity profiles
- Compute statistics on the data
- Filter and resample data

## Installation

### Prerequisites

- Python 3.6+
- ROS2 (with rosbag2_py, rclpy, and rosidl_runtime_py)

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python analyze.py /path/to/your/bagfile
```

This will print information about the bag file, including available topics, message counts, and duration.

### Specifying Topics

```bash
python analyze.py /path/to/your/bagfile --topics /cmd_vel /scan /odom
```

This will print information about the specified topics, including available fields and message counts.

### Example Script

The `example.py` script demonstrates how to use the tools:

```bash
python example.py /path/to/your/bagfile --show-plots
```

This will:
1. Print information about the bag file
2. Create visualizations for the data
3. Show examples of accessing message fields

## Project Structure

- `analyze.py`: Main script with the BagReader class for reading and analyzing ROS2 bag files
- `visualization.py`: Functions for creating visualizations
- `example.py`: Example script demonstrating how to use the tools
- `requirements.txt`: List of Python dependencies

## Accessing Message Fields

You can access fields in ROS2 messages using dot notation:

```python
# Create a BagReader instance
reader = BagReader("/path/to/your/bagfile")

# Get the x position from an Odometry message
position_x_values = reader.get_message_field_values("/odom", "pose.pose.position.x")

# Get the linear velocity from a Twist message
linear_vel_values = reader.get_message_field_values("/cmd_vel", "linear.x")

# Get the ranges from a LaserScan message
ranges_values = reader.get_message_field_values("/scan", "ranges")
```

## Creating Visualizations

You can create visualizations for the data:

```python
# Create a BagReader instance
reader = BagReader("/path/to/your/bagfile")

# Create a time series plot for a specific field
plot_field_time_series(reader, "/odom", "pose.pose.position.x")

# Create a trajectory plot
plot_trajectory(reader, "/odom")

# Create a velocity profile plot
plot_velocity_profile(reader, "/cmd_vel", "linear.x", "angular.z")

# Create a dashboard of plots
create_dashboard(reader, "output_dir")
```

## Extending the Tool

### Adding New Visualizations

To add a new visualization, add a function in `visualization.py`:

```python
def your_new_plot(bag_reader: BagReader, topic: str, field_path: str,
                 title: str, 
                 output_path: Optional[Union[str, Path]] = None,
                 show_plot: bool = True) -> plt.Figure:
    """
    Create a new type of plot.
    
    Args:
        bag_reader: BagReader instance
        topic: Topic name
        field_path: Path to the field to plot
        title: Plot title
        output_path: Path to save the plot to
        show_plot: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    # Get the field values
    values = bag_reader.get_message_field_values(topic, field_path)
    
    # Extract values and timestamps
    field_values = np.array([v for v, _ in values])
    timestamps = np.array([t for _, t in values])
    
    # Create your plot
    # ...
    
    return fig
```

Then, update the `create_dashboard` function to use your new visualization if needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 