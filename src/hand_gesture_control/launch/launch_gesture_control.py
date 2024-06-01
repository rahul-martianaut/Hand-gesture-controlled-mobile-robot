from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hand_gesture_control',
            executable='hand_gesture_node.py',
            name='hand_gesture_node',
            output='screen',
          
        )
    ])

