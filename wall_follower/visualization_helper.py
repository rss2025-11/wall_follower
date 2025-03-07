#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

class WallFollowerVisualizer:
    """
    Helper class for visualizing wall follower data in RViz.
    """
    
    def __init__(self, node: Node):
        """
        Initialize the visualizer with the parent node.
        
        Args:
            node: The ROS node that will publish the visualization markers
        """
        self.node = node
        
        # Create publishers for visualization markers
        self.view_angle_pub = node.create_publisher(
            MarkerArray, '/wall_follower/view_angle', 10
        )
        self.wall_line_pub = node.create_publisher(
            Marker, '/wall_follower/wall_line', 10
        )
        self.desired_line_pub = node.create_publisher(
            Marker, '/wall_follower/desired_line', 10
        )
        
        # Standard colors
        self.red = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        self.green = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        self.blue = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        self.yellow = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        
        # Marker lifetime (5 seconds)
        self.marker_lifetime = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
    
    def visualize_view_angle(self, angle_front_margin, angle_back_margin, side):
        """
        Visualize the view angle used for wall following.
        
        Args:
            angle_front_margin: Front margin angle in radians
            angle_back_margin: Back margin angle in radians
            side: Side to follow (-1 for right, 1 for left)
        """
        marker_array = MarkerArray()
        
        # Calculate the actual angles based on side
        if side == -1:  # Right side
            lower_angle = -angle_back_margin
            upper_angle = -angle_front_margin
        else:  # Left side
            lower_angle = angle_front_margin
            upper_angle = angle_back_margin
        
        # Create markers for the view angle boundaries
        for i, angle in enumerate([lower_angle, upper_angle]):
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = "view_angle"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.lifetime = self.marker_lifetime
            
            # Create a line from origin extending 5 meters in the angle direction
            start_point = Point(x=0.0, y=0.0, z=0.0)
            end_point = Point(
                x=5.0 * np.cos(angle),
                y=5.0 * np.sin(angle),
                z=0.0
            )
            
            marker.points = [start_point, end_point]
            marker.scale.x = 0.05  # Line width
            marker.color = self.yellow
            
            marker_array.markers.append(marker)
        
        self.view_angle_pub.publish(marker_array)
    
    def visualize_wall_line(self, m, b, x_range=(-5.0, 5.0), num_points=50):
        """
        Visualize the line of best fit representing the wall.
        
        Args:
            m: Slope of the line
            b: Y-intercept of the line
            x_range: Range of x values to plot
            num_points: Number of points to use for the line
        """
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "wall_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = self.marker_lifetime
        
        # Generate points along the line
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        points = []
        
        for x in x_values:
            y = m * x + b
            points.append(Point(x=float(x), y=float(y), z=0.0))
        
        marker.points = points
        marker.scale.x = 0.05  # Line width
        marker.color = self.red
        
        self.wall_line_pub.publish(marker)
    
    def visualize_desired_line(self, m, b, desired_distance, side, x_range=(-5.0, 5.0), num_points=50):
        """
        Visualize a line parallel to the wall at the desired distance.
        
        Args:
            m: Slope of the wall line
            b: Y-intercept of the wall line
            desired_distance: Desired distance from the wall
            side: Side to follow (-1 for right, 1 for left)
            x_range: Range of x values to plot
            num_points: Number of points to use for the line
        """
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "desired_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = self.marker_lifetime
        
        # Calculate the parallel line at the desired distance
        # For a line y = mx + b, the perpendicular distance d from a point (x0, y0) is:
        # d = |y0 - mx0 - b| / sqrt(1 + m^2)
        # To create a parallel line at distance d, we adjust the y-intercept:
        # b_new = b ± d * sqrt(1 + m^2)
        # The sign depends on which side of the line we want
        
        # Adjust the y-intercept for the desired distance
        adjustment = desired_distance * np.sqrt(1 + m**2)
        # Side determines which direction to adjust (positive or negative)
        b_new = b - side * adjustment
        
        # Generate points along the parallel line
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        points = []
        
        for x in x_values:
            y = m * x + b_new
            points.append(Point(x=float(x), y=float(y), z=0.0))
        
        marker.points = points
        marker.scale.x = 0.05  # Line width
        marker.color = self.green
        
        self.desired_line_pub.publish(marker)