# launch_simulation.py
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_description')
    sdf = os.path.join(pkg_share, 'sdf', 'robot.sdf')
    world = os.path.join(pkg_share, 'world', 'jetbot_demo.world')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'],
            output='screen'),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robot', '-file', sdf],
            output='screen'),
    ])

