"""ROS 2 node that deploys the Isaac Lab obstacle-avoidance policy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .policy_contract import (
    POLICY_ACTIONS,
    POLICY_OBSERVATIONS,
    PolicyLimits,
    action_to_command,
    build_observation,
    resample_laser_scan,
    wrap_angle,
)
from .pose_sources import POSE_SOURCE_TYPES, PoseMeasurement


class PolicyController(Node):
    """Run the exported policy against ROS sensor and goal messages."""

    def __init__(self) -> None:
        super().__init__("rl_goal_controller")
        self._declare_parameters()

        self._limits = PolicyLimits(
            lidar_max_range=float(self.get_parameter("lidar_max_range").value),
            max_linear_speed=float(self.get_parameter("max_linear_speed").value),
            max_angular_speed=float(self.get_parameter("max_angular_speed").value),
        )
        self._validate_parameters()
        self._session = self._load_policy()

        self._enabled = bool(self.get_parameter("enabled").value)
        self._pose: PoseMeasurement | None = None
        self._pose_received_at: float | None = None
        self._previous_pose: PoseMeasurement | None = None
        self._previous_pose_time: float | None = None
        self._linear_velocity = 0.0
        self._angular_velocity = 0.0
        self._scan: np.ndarray | None = None
        self._closest_scan_return = math.inf
        self._scan_received_at: float | None = None
        self._goal: tuple[float, float] | None = None
        self._goal_frame = ""
        self._goal_was_reached = False
        self._previous_action = np.zeros(POLICY_ACTIONS, dtype=np.float32)
        self._command_linear = 0.0
        self._command_angular = 0.0
        self._last_control_time: float | None = None
        self._last_status = ""
        self._last_warning: dict[str, float] = {}

        self._cmd_publisher = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10
        )
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._reached_publisher = self.create_publisher(Bool, "~/goal_reached", state_qos)
        self._status_publisher = self.create_publisher(String, "~/status", state_qos)
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self._goal_callback,
            10,
        )

        source_name = str(self.get_parameter("pose_source_type").value)
        source = POSE_SOURCE_TYPES[source_name]
        self._pose_adapter = source
        self.create_subscription(
            source.message_type,
            str(self.get_parameter("pose_topic").value),
            self._pose_callback,
            10,
        )
        velocity_topic = str(self.get_parameter("velocity_topic").value)
        if velocity_topic and not (
            source_name == "odometry" and velocity_topic == str(self.get_parameter("pose_topic").value)
        ):
            self.create_subscription(Odometry, velocity_topic, self._velocity_callback, 10)

        self.create_service(SetBool, "~/enable", self._enable_callback)
        frequency = float(self.get_parameter("control_frequency").value)
        self._timer = self.create_timer(1.0 / frequency, self._control_callback)
        self._publish_reached(False)
        self._set_status("waiting_for_goal")
        self.get_logger().info(
            f"Policy ready; pose={source_name}:{self.get_parameter('pose_topic').value}, "
            f"scan={self.get_parameter('scan_topic').value}, cmd={self.get_parameter('cmd_vel_topic').value}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("pose_source_type", "odometry")
        self.declare_parameter("pose_topic", "/odom")
        self.declare_parameter("velocity_topic", "")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_frequency", 20.0)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("scan_timeout", 0.5)
        self.declare_parameter("goal_tolerance", 0.2)
        self.declare_parameter("emergency_stop_distance", 0.15)
        # The A1's laser-frame +x axis points toward its cable. On this robot
        # the cable faces rearward, so laser +x is rotated pi from base +x.
        self.declare_parameter("laser_yaw_offset", math.pi)
        self.declare_parameter("lidar_max_range", 8.0)
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", .5)
        self.declare_parameter("max_linear_accel", 1.5)
        self.declare_parameter("max_angular_accel", 1.0)
        self.declare_parameter("enabled", True)

    def _validate_parameters(self) -> None:
        source_name = str(self.get_parameter("pose_source_type").value)
        if source_name not in POSE_SOURCE_TYPES:
            choices = ", ".join(sorted(POSE_SOURCE_TYPES))
            raise ValueError(f"pose_source_type must be one of: {choices}")
        for name in (
            "control_frequency",
            "pose_timeout",
            "scan_timeout",
            "goal_tolerance",
            "emergency_stop_distance",
            "lidar_max_range",
            "max_linear_speed",
            "max_angular_speed",
            "max_linear_accel",
            "max_angular_accel",
        ):
            if float(self.get_parameter(name).value) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _load_policy(self):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required; install the ROS/pip package before starting this node"
            ) from error

        configured_path = str(self.get_parameter("model_path").value)
        model_path = (
            Path(configured_path).expanduser()
            if configured_path
            else Path(get_package_share_directory("rl_controller")) / "models" / "policy.onnx"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"policy model not found: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or (isinstance(inputs[0].shape[-1], int) and inputs[0].shape[-1] != POLICY_OBSERVATIONS):
            raise RuntimeError(f"expected one {POLICY_OBSERVATIONS}-element model input, got {inputs}")
        if not outputs:
            raise RuntimeError("policy model has no outputs")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.get_logger().info(f"Loaded policy model from {model_path}")
        return session

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _scan_callback(self, message: LaserScan) -> None:
        try:
            self._scan, self._closest_scan_return = resample_laser_scan(
                message.ranges,
                message.angle_min,
                message.angle_increment,
                message.range_min,
                message.range_max,
                self._limits.lidar_max_range,
                float(self.get_parameter("laser_yaw_offset").value),
            )
            self._scan_received_at = self._now()
        except ValueError as error:
            self._warn_throttled("scan", f"Ignoring invalid LaserScan: {error}")

    def _pose_callback(self, message) -> None:
        now = self._now()
        try:
            pose = self._pose_adapter.extract(message)
        except ValueError as error:
            self._warn_throttled("pose", f"Ignoring invalid pose: {error}")
            return

        if pose.linear_velocity is not None:
            self._linear_velocity = pose.linear_velocity
            self._angular_velocity = pose.angular_velocity or 0.0
        elif self._previous_pose is not None and self._previous_pose_time is not None:
            dt = now - self._previous_pose_time
            if 1.0e-3 < dt < 2.0:
                dx = pose.x - self._previous_pose.x
                dy = pose.y - self._previous_pose.y
                self._linear_velocity = (
                    math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
                ) / dt
                self._angular_velocity = wrap_angle(pose.yaw - self._previous_pose.yaw) / dt
        self._previous_pose = pose
        self._previous_pose_time = now
        self._pose = pose
        self._pose_received_at = now

    def _velocity_callback(self, message: Odometry) -> None:
        self._linear_velocity = message.twist.twist.linear.x
        self._angular_velocity = message.twist.twist.angular.z

    def _goal_callback(self, message: PoseStamped) -> None:
        self._goal = (message.pose.position.x, message.pose.position.y)
        self._goal_frame = message.header.frame_id
        self._goal_was_reached = False
        self._previous_action.fill(0.0)
        self._command_linear = 0.0
        self._command_angular = 0.0
        self._publish_reached(False)
        self._set_status("goal_active")
        self.get_logger().info(
            f"Accepted goal ({self._goal[0]:.3f}, {self._goal[1]:.3f}) in frame '{self._goal_frame}'"
        )

    def _enable_callback(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self._enabled = request.data
        if not self._enabled:
            self._stop("disabled", reset_action=True)
        response.success = True
        response.message = "controller enabled" if self._enabled else "controller disabled and stopped"
        return response

    def _control_callback(self) -> None:
        now = self._now()
        if not self._enabled:
            self._stop("disabled")
            return
        if self._goal is None:
            self._stop("goal_reached" if self._goal_was_reached else "waiting_for_goal")
            return
        if self._pose is None or self._pose_received_at is None:
            self._stop("waiting_for_pose")
            return
        if now - self._pose_received_at > float(self.get_parameter("pose_timeout").value):
            self._stop("pose_stale", reset_action=True)
            return
        if self._scan is None or self._scan_received_at is None:
            self._stop("waiting_for_scan")
            return
        if now - self._scan_received_at > float(self.get_parameter("scan_timeout").value):
            self._stop("scan_stale", reset_action=True)
            return
        if self._goal_frame and self._pose.frame_id and self._goal_frame != self._pose.frame_id:
            self._stop("frame_mismatch", reset_action=True)
            self._warn_throttled(
                "frame",
                f"Goal frame '{self._goal_frame}' does not match pose frame '{self._pose.frame_id}'",
            )
            return

        distance = math.hypot(self._goal[0] - self._pose.x, self._goal[1] - self._pose.y)
        if distance <= float(self.get_parameter("goal_tolerance").value):
            self._goal = None
            self._goal_was_reached = True
            self._stop("goal_reached", reset_action=True)
            self._publish_reached(True)
            self.get_logger().info("Goal reached")
            return
        if self._closest_scan_return < float(self.get_parameter("emergency_stop_distance").value):
            self._stop("emergency_stop", reset_action=True)
            self._warn_throttled(
                "obstacle", f"Obstacle at {self._closest_scan_return:.3f} m; command held at zero"
            )
            return

        observation = build_observation(
            self._scan,
            self._pose.x,
            self._pose.y,
            self._pose.yaw,
            self._goal[0],
            self._goal[1],
            self._linear_velocity,
            self._angular_velocity,
            self._previous_action,
            self._limits,
        )
        
        
        output = self._session.run([self._output_name], {self._input_name: observation[None, :]})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.size != POLICY_ACTIONS or not np.all(np.isfinite(action)):
            self._stop("invalid_policy_output", reset_action=True)
            self._warn_throttled("policy", f"Invalid policy output: {action}")
            return
        action = np.clip(action, -1.0, 1.0)
        target_linear, target_angular = action_to_command(action, self._limits)
        dt = 1.0 / float(self.get_parameter("control_frequency").value)
        if self._last_control_time is not None:
            dt = min(max(now - self._last_control_time, 0.0), 0.25)
        self._last_control_time = now
        self._command_linear = self._move_toward(
            self._command_linear,
            target_linear,
            float(self.get_parameter("max_linear_accel").value) * dt,
        )
        self._command_angular = self._move_toward(
            self._command_angular,
            target_angular,
            float(self.get_parameter("max_angular_accel").value) * dt,
        )
        self._previous_action = action
        self._publish_command(self._command_linear, self._command_angular)
        self._set_status("driving")

    @staticmethod
    def _move_toward(current: float, target: float, maximum_delta: float) -> float:
        return current + max(-maximum_delta, min(maximum_delta, target - current))

    def _publish_command(self, linear: float, angular: float) -> None:
        message = Twist()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self._cmd_publisher.publish(message)

    def _stop(self, status: str, reset_action: bool = False) -> None:
        self._command_linear = 0.0
        self._command_angular = 0.0
        self._last_control_time = None
        if reset_action:
            self._previous_action.fill(0.0)
        self._publish_command(0.0, 0.0)
        self._set_status(status)

    def _publish_reached(self, reached: bool) -> None:
        message = Bool()
        message.data = reached
        self._reached_publisher.publish(message)

    def _set_status(self, status: str) -> None:
        if status == self._last_status:
            return
        self._last_status = status
        message = String()
        message.data = status
        self._status_publisher.publish(message)

    def _warn_throttled(self, key: str, message: str, period: float = 2.0) -> None:
        now = self._now()
        if now - self._last_warning.get(key, -math.inf) >= period:
            self.get_logger().warning(message)
            self._last_warning[key] = now

    def destroy_node(self) -> bool:
        if hasattr(self, "_cmd_publisher"):
            self._publish_command(0.0, 0.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            rclpy.logging.get_logger("rl_goal_controller").fatal(str(error))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
