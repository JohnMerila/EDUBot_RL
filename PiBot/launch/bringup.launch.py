from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include_launch(package_name, launch_file):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare(package_name), "launch", launch_file]
            )
        )
    )


def generate_launch_description():
    return LaunchDescription(
        [
            include_launch("PiBot", "PiBot_launch.py"),
            include_launch("sllidar_ros2", "sllidar_a1_launch.py"),
            include_launch("rl_controller", "rf2o_controller.launch.py"),
        ]
    )
