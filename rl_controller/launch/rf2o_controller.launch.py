from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    controller_config = (
        get_package_share_directory("rl_controller") + "/config/rf2o.yaml"
    )

    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    scan_frame = LaunchConfiguration("scan_frame")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    publish_laser_tf = LaunchConfiguration("publish_laser_tf")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf")

    return LaunchDescription(
        [
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/odom_rf2o"),
            DeclareLaunchArgument("scan_frame", default_value="laser"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("publish_laser_tf", default_value="true"),
            DeclareLaunchArgument("publish_odom_tf", default_value="true"),
            # Defaults reproduce the LiDAR placement used during policy training.
            # Measure the physical robot and override these values if needed.
            DeclareLaunchArgument("laser_x", default_value="0.08"),
            DeclareLaunchArgument("laser_y", default_value="0.0"),
            DeclareLaunchArgument("laser_z", default_value="0.215"),
            DeclareLaunchArgument("laser_roll", default_value="0.0"),
            DeclareLaunchArgument("laser_pitch", default_value="0.0"),
            DeclareLaunchArgument("laser_yaw", default_value="3.141592653589793"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_laser_tf",
                condition=IfCondition(publish_laser_tf),
                arguments=[
                    "--x", LaunchConfiguration("laser_x"),
                    "--y", LaunchConfiguration("laser_y"),
                    "--z", LaunchConfiguration("laser_z"),
                    "--roll", LaunchConfiguration("laser_roll"),
                    "--pitch", LaunchConfiguration("laser_pitch"),
                    "--yaw", LaunchConfiguration("laser_yaw"),
                    "--frame-id", base_frame,
                    "--child-frame-id", scan_frame,
                ],
                output="screen",
            ),
            Node(
                package="rf2o_laser_odometry",
                executable="rf2o_laser_odometry_node",
                name="rf2o_laser_odometry",
                output="screen",
                parameters=[
                    {
                        "laser_scan_topic": scan_topic,
                        "odom_topic": odom_topic,
                        "publish_tf": ParameterValue(publish_odom_tf, value_type=bool),
                        "base_frame_id": base_frame,
                        "odom_frame_id": odom_frame,
                        "init_pose_from_topic": "",
                        "freq": 20.0,
                    }
                ],
            ),
            Node(
                package="rl_controller",
                executable="policy_controller",
                name="rl_goal_controller",
                output="screen",
                parameters=[
                    controller_config,
                    {
                        "pose_topic": odom_topic,
                        "scan_topic": scan_topic,
                        "laser_yaw_offset": ParameterValue(
                            LaunchConfiguration("laser_yaw"), value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
