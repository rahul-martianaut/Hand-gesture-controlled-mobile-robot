# Hand Gesture Controlled Mobile Robot

This project involves the control of a mobile robot using hand gestures. The hand gestures are recognized using the MediaPipe and OpenCV libraries implemented in PyTorch. For the navigation of the robot, ROS2 is used. Visualization is done using Rviz2 and Gazebo. The SLAM (Simultaneous Localization and Mapping) is handled by the SLAM Toolbox. The mobile robot used in this project is the NVIDIA Jetbot.

![vid](https://github.com/rahul-martianaut/Hand-gesture-controlled-mobile-robot/assets/117083668/a5a958d1-4bbd-4e74-9728-2e1ea8e0f284)

## Requirements
- Ubuntu 20.04
- ROS Foxy
- Gazebo
- Rviz2
- MediaPipe 0.10.11
- OpenCV 3.4.2 or Later
- Pytorch

## Installation

1. Clone this repository:
    ```bash
    cd ~/hand-gesture_ws/src
    git clone https://github.com/rahul-martianaut/Hand-gesture-controlled-mobile-robot.git
    ```

2. Clone the `jetbot_ros` package:
    ```bash
    git clone https://github.com/dusty-nv/jetbot_ros
    ```

3. Replace the original `gazebo_world.launch.py` file inside `jetbot_ros/launch` with the `gazebo_world.launch.py` file inside the package `robot_control/launch`. This will launch Gazebo with the robot in the world and enable hand gesture recognition.

## Usage

### Launching the Simulation World

1. Launch the Gazebo simulation world:
    ```bash
    ros2 launch jetbot_ros gazebo_world.launch.py
    ```

### Launching SLAM

1. Open another terminal and run the following command to launch SLAM:
    ```bash
    ros2 launch jetbot_navigation slam_launch.py
    ```

### Visualization with Rviz2

1. Open another terminal and run the following command to visualize with Rviz2:
    ```bash
    ros2 run rviz2 rviz2
    ```

## Hand Gestures

The hand gestures used to control the robot are:
- **0**: Move forward
- **1**: Stop movement
- **2**: Turn right
- **3**: Turn left

The hand gesture recognition model can be trained with custom gestures using the script `hand_gesture_control/scripts/train.py`

---

Feel free to contribute to this project.

Happy Coding!
