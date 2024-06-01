#!/usr/bin/env python
# robot_control_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.subscription = self.create_subscription(
            String,
            'gesture',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Twist, 'jetbot/cmd_vel', 10)

    def listener_callback(self, msg):
        gesture = msg.data
        #print(gesture)
        twist = Twist()
        if gesture == '0':
            twist.linear.x = 0.5
        elif gesture == '1':
            twist.linear.x = 0.0
        elif gesture == '2':
            twist.angular.z = 0.5
        elif gesture == '3':
            twist.angular.z = -0.5
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.publisher_.publish(twist)
	

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
