#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32
from typing import Dict
from wall_follower.visualization_helper import WallFollowerVisualizer


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")  # string
        self.declare_parameter("side", 1)  # integer (-1 for right, 1 for left)
        self.declare_parameter("velocity", 1.0)  # double
        self.declare_parameter("desired_distance", 1.0)  # double
        self.declare_parameter("lookahead_ratio", 5.0)  # double
        self.declare_parameter("kp", 2.0)  # double
        self.declare_parameter("kd", 1.0)  # double
        # Fetch constants from the ROS parameter server
        # This is necessary for the tests to be able to test varying parameters!
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.SCAN_TOPIC = (
            self.get_parameter("scan_topic").get_parameter_value().string_value
        )
        self.DRIVE_TOPIC = (
            self.get_parameter("drive_topic").get_parameter_value().string_value
        )
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = (
            self.get_parameter("velocity").get_parameter_value().double_value
        )
        self.DESIRED_DISTANCE = (
            self.get_parameter("desired_distance").get_parameter_value().double_value
        )
        self.LOOKAHEAD_RATIO = (
            self.get_parameter("lookahead_ratio").get_parameter_value().double_value
        )
        self.KP = self.get_parameter("kp").get_parameter_value().double_value
        self.KD = self.get_parameter("kd").get_parameter_value().double_value

        # Define angle ranges for scan filtering
        self.ANGLE_FRONT_MARGIN = -np.pi / 12  # in radians
        self.ANGLE_BACK_MARGIN = np.pi / 3

        self.front_distance = 0.0

        # Initialize publishers and subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.parse_scan_data, 1
        )

        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, self.DRIVE_TOPIC, 10
        )

        self.front_distance_subscriber = self.create_subscription(
            Float32,
            "/safety_controller/distance_to_front_wall",
            self.front_distance_callback,
            10,
        )

        self.visualizer = WallFollowerVisualizer(self)

    def parse_scan_data(self, scan_data: LaserScan) -> None:
        """
        Take in the current scan data and compute the distance and angle to the wall
        for the PD controller.

        Returns early with a default dict if there are not enough valid points.
        """
        self.visualizer.visualize_view_angle(
            self.ANGLE_FRONT_MARGIN, self.ANGLE_BACK_MARGIN, self.SIDE
        )
        # Get angles and ranges from scan data
        angles = np.arange(
            scan_data.angle_min, scan_data.angle_max, scan_data.angle_increment
        )
        ranges = np.array(scan_data.ranges)

        # Filter points based on side and angle.
        # Note that positive is ccw (to the left) and negative is cw (to the right)
        if self.SIDE == -1:  # Right side
            lower_angle = -self.ANGLE_BACK_MARGIN
            upper_angle = -self.ANGLE_FRONT_MARGIN
        else:  # Left side
            lower_angle = self.ANGLE_FRONT_MARGIN
            upper_angle = self.ANGLE_BACK_MARGIN

        mask = (angles >= lower_angle) & (angles <= upper_angle)
        angles = angles[mask]
        ranges = ranges[mask]

        # Convert from polar to cartesian coordinates.
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Include only points within a lookahead distance.
        max_distance = self.LOOKAHEAD_RATIO * self.DESIRED_DISTANCE
        distance_mask = np.sqrt(x**2 + y**2) <= max_distance
        x = x[distance_mask]
        y = y[distance_mask]
        angles = angles[distance_mask]

        if len(x) < 2:  # Need at least 2 points for line fitting
            return {"error": 0, "angle_to_wall": 0.0}

        # if front distance is leq desired distance, weight front points more
        if self.front_distance <= self.DESIRED_DISTANCE:
            # Find points within front margin
            front_mask = np.abs(angles) <= abs(self.ANGLE_FRONT_MARGIN)
            # Get the front points
            front_x = x[front_mask]
            front_y = y[front_mask]
            # Duplicate front points by concatenating them with all points
            x = np.concatenate([x, front_x])
            y = np.concatenate([y, front_y])

        m, b = self.fit_line(x, y)

        # Calculate the wall's orientation and perpendicular distance.
        angle_to_wall = np.arctan(m)
        wall_distance = abs(b) / np.sqrt(m**2 + 1)

        self.visualizer.visualize_wall_line(m, b)
        self.visualizer.visualize_desired_line(m, b, self.DESIRED_DISTANCE, self.SIDE)

        scan_result = {
            "distance_to_wall": wall_distance,
            "angle_to_wall": angle_to_wall,
        }

        self.control_car(scan_result)

    def fit_line(self, x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, b

    def control_car(self, scan_result: Dict[str, float]) -> None:
        """
        Uses PD control based on the parsed scan data to compute a steering command.
        The error is defined so that negative means too close to the wall and positive means
        too far away.
        """
        error = (
            self.DESIRED_DISTANCE - scan_result["distance_to_wall"]
        )  # negative if too close and positive if too far

        # Compute derivative of the error
        if hasattr(self, "prev_error"):
            derivative = error - self.prev_error
        else:
            derivative = 0.0  # No derivative on the first measurement.
        self.prev_error = error

        # PD control to compute steering.
        steering = self.KP * error + self.KD * derivative

        # mulitply by SIDE to get correct steering direction
        steering = -self.SIDE * steering

        steering = max(min(steering, np.pi / 8), -np.pi / 8)

        control_data = {
            "steering_angle": steering,
            "speed": self.VELOCITY,
        }
        self.publish_control(control_data)

    def publish_control(self, control_data):
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.steering_angle = control_data["steering_angle"]
        ackermann_msg.drive.speed = control_data["speed"]
        self.drive_publisher.publish(ackermann_msg)

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!

        This is used by the test cases to modify the parameters during testing.
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == "side":
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == "velocity":
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == "desired_distance":
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(
                    f"Updated desired_distance to {self.DESIRED_DISTANCE}"
                )
        return SetParametersResult(successful=True)

    def front_distance_callback(self, distance: Float32):
        self.front_distance = distance.data


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
