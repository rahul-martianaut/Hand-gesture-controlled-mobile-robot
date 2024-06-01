import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

from ament_index_python.packages import get_package_share_directory
 
def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
     
    robot_name = DeclareLaunchArgument('robot_name', default_value='jetbot')
    robot_model = DeclareLaunchArgument('robot_model', default_value='simple_diff_ros')  # jetbot_ros
    
    robot_x = DeclareLaunchArgument('x', default_value='0.0') #-0.3
    robot_y = DeclareLaunchArgument('y', default_value='0.0') #-2.65
    robot_z = DeclareLaunchArgument('z', default_value='0.0')

    hand_gesture_pkg_dir = get_package_share_directory('hand_gesture_control')
    robot_control_pkg_dir = get_package_share_directory('robot_control')
    
    world_file_name = 'maze.world'
    pkg_dir = get_package_share_directory('jetbot_ros')
 
    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')
 
    world = os.path.join(pkg_dir, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_dir, 'launch')

    hand_gesture_node = Node(
        package='hand_gesture_control',
        executable='hand_gesture_node.py',
        name='hand_gesture_node',
        output='screen'
    )

    robot_control_node = Node(
        package='robot_control',
        executable='robot_control_node.py',
        name='robot_control_node',
        output='screen'
    )

    map_to_odom_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
 
    gazebo = ExecuteProcess(
                cmd=['gazebo', '--verbose', world, 
                     '-s', 'libgazebo_ros_init.so', 
                     '-s', 'libgazebo_ros_factory.so',
                     '-g', 'libgazebo_user_camera_control_system.so'],
                output='screen', emulate_tty=True)

    
    spawn_entity = Node(package='jetbot_ros', node_executable='gazebo_spawn',   # FYI 'node_executable' is renamed to 'executable' in Foxy
                        parameters=[
                            {'name': LaunchConfiguration('robot_name')},
                            {'model': LaunchConfiguration('robot_model')},
                            {'x': LaunchConfiguration('x')},
                            {'y': LaunchConfiguration('y')},
                            {'z': LaunchConfiguration('z')},
                        ],
                        output='screen', emulate_tty=True)
                        
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
	arguments=["0","0","0","0","0","0","chassis","base_link"]
	)
 
    return LaunchDescription([
        robot_name,
        robot_model,
        robot_x,
        robot_y,
        robot_z,
        hand_gesture_node,
        robot_control_node,
        map_to_odom_publisher,
        gazebo,
        spawn_entity,
        static_broadcaster,
        static_broadcaster_2,
    ])
