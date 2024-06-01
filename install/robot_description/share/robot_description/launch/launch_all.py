from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    robot_description_pkg_share = get_package_share_directory('robot_description')
    hand_gesture_pkg_share = get_package_share_directory('hand_gesture_control')
    robot_control_pkg_share = get_package_share_directory('robot_control')
    
    urdf = os.path.join(robot_description_pkg_share, 'sdf', 'robot.urdf')
    world = os.path.join(robot_description_pkg_share, 'world', 'jetbot_demo.world')
    rviz_config = os.path.join(robot_description_pkg_share, 'rviz', 'stage.rviz')

    # Include launch files from other packages
    launch_gesture_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(hand_gesture_pkg_share, 'launch', 'launch_gesture_control.py'))
    )
    
    launch_robot_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(robot_control_pkg_share, 'launch', 'launch_robot_control.py'))
    )

    # Gazebo launch
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world],
        output='screen'
    )

    # Robot spawn
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'robot', '-file', urdf],
        output='screen'
    )

    # RViz launch
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    # Adding robot state publisher and joint state publisher from robot_description
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': open(urdf).read()}]
    )
    
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'source_list': ['/robot/joint_states']}]
    )

    static_broadcaster = Node(
	package="tf2_ros",
	executable="static_transform_publisher",
	output="screen",
	arguments=["0","0","0","0","0","0","chassis","lidar_link"]
	)
    static_broadcaster_2 = Node(
	package="tf2_ros",
	executable="static_transform_publisher",
	output="screen",
	arguments=["0","0","0","0","0","0","chassis","odom"]
	)

    return LaunchDescription([
        gazebo,
        spawn_robot,
        # rviz,
        launch_gesture_control,
        launch_robot_control,
        robot_state_publisher,
        joint_state_publisher,
        static_broadcaster,
        static_broadcaster_2
    ])

