#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult


class WallFollower(Node):
    # WALL_TOPIC = "/wall"

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS! 
        # self.declare_parameter("scan_topic", "default")
        # self.declare_parameter("drive_topic", "default")
        # self.declare_parameter("side", 1)
        # self.declare_parameter("velocity", 1.0)
        # self.declare_parameter("desired_distance", 1.0)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        # self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        # self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        # self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        # self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        # self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
		
        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS! 
        self.add_on_set_parameters_callback(self.parameters_callback)
  
        self.previous_error = 0.0

        # TODO: Initialize your publishers and subscribers here
        self.laser_scan_sub = self.create_subscription(LaserScan,
                                                     "/scan",
                                                     self.listener_callback,
                                                     10)
        self.publish_ackermann_cmd = self.create_publisher(AckermannDriveStamped,
                                                           "/drive",
                                                        #    "/vesc/high_level/input/nav_0",
                                                           10)
        # self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)
    # TODO: Write your callback functions here    
    def listener_callback(self, LaserScanMsg):
        # manipulate laserscan message based on parameters
        self.get_logger().info("started the callback")
        ranges = np.array(LaserScanMsg.ranges)
        angle_min = LaserScanMsg.angle_min
        angle_max = LaserScanMsg.angle_max
        angle_increment = LaserScanMsg.angle_increment
        angles = np.arange(angle_min, angle_max, angle_increment)
        if self.SIDE == 1:
            # fix the range for which the laser scan data is being used
            mask = (angles>=0*np.pi/6) & (angles<=np.pi/2) # was 6 and 3
        else:
            mask = (angles<=0*-np.pi/6) & (angles>=-np.pi/2)
        mask = mask & (ranges < 4)
        relavent_ranges = ranges[mask]
        relavent_angles = angles[mask]

        # find the equation of the wall's line
        # use some form of least squares...
        x = relavent_ranges*np.cos(relavent_angles)
        y = relavent_ranges*np.sin(relavent_angles)
        A = np.vstack([x, np.ones_like(x)]).T  # coefficient matrix
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve Ax = b
        x_ls = x
        y_ls = m*x+b
        # Convert y = mx + b to Ax + By + C = 0 form
        A = -m
        B = 1
        C = -b
        x_r, y_r = 0, 0 # robot is center of coordinate system
        distance_from_wall = np.abs(A*x_r+B*y_r+C)/np.sqrt(A**2+B**2)
        # distance_from_wall = relavent_ranges
        distance_error = self.DESIRED_DISTANCE - distance_from_wall 

        
        Kp = 0.5
        Kd = 0.1 # was 3

        if self.VELOCITY > 1:
            Kp = 0.7
            Kd = 4 # was 1
        if self.VELOCITY > 2:
            Kp = 4 # was 1.7
            Kd = 9 # was 2
            distance_error = distance_error + 0.075
            
        derivative = distance_error - self.previous_error

        control_output = Kp * distance_error + Kd * derivative
        acker_cmd = AckermannDriveStamped()
        acker_cmd.header.stamp = self.get_clock().now().to_msg()
        acker_cmd.header.frame_id = 'map'
        acker_cmd.drive.steering_angle = float(-self.SIDE*control_output)
        acker_cmd.drive.steering_angle_velocity = 0.0 # it was 2
        acker_cmd.drive.speed = self.VELOCITY
        acker_cmd.drive.acceleration = 0.0 # it was 0.5
        acker_cmd.drive.jerk = 0.0

        self.previous_error = distance_error
        
        acker_cmd = AckermannDriveStamped()
        acker_cmd.header.stamp = self.get_clock().now().to_msg()
        acker_cmd.header.frame_id = 'map'
        acker_cmd.drive.steering_angle = 0.0
        acker_cmd.drive.steering_angle_velocity = 0.0 # it was 2
        acker_cmd.drive.speed = 1.0
        acker_cmd.drive.acceleration = 0.0 # it was 0.5
        acker_cmd.drive.jerk = 0.0
        self.get_logger().info("Reached pre-publishing")
        self.publish_ackermann_cmd.publish(acker_cmd)
        # VisualizationTools.plot_line(x_ls, y_ls, self.line_pub, frame="/laser")

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!
        
        This is used by the test cases to modify the parameters during testing. 
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    