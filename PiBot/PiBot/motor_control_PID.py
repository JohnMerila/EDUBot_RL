"""Closed-loop differential-drive motor controller for PiBot.

The encoder_data node currently publishes wheel angles in degrees in the
JointState position field. This node unwraps those angles, converts measured
wheel velocities to rad/s, and regulates them to targets calculated from
cmd_vel.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from gpiozero import Motor
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrapped_degree_delta(current: float, previous: float) -> float:
    """Return the shortest signed encoder change across the 0/360 boundary."""
    return (current - previous + 180.0) % 360.0 - 180.0


def move_toward(current: float, target: float, maximum_delta: float) -> float:
    """Move current toward target without changing by more than maximum_delta."""
    return current + clamp(target - current, -maximum_delta, maximum_delta)


class PIDController:
    """PID with feed-forward, output limiting, and conditional integration."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        feedforward: float,
        integral_limit: float,
        output_limit: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.feedforward = feedforward
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, target: float, measured: float, dt: float) -> float:
        error = target - measured
        derivative = 0.0
        if self.previous_error is not None and dt > 0.0:
            derivative = (error - self.previous_error) / dt

        candidate_integral = clamp(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        unconstrained = (
            self.feedforward * target
            + self.kp * error
            + self.ki * candidate_integral
            + self.kd * derivative
        )
        # Do not actively reverse a motor merely to brake an overspeed wheel.
        # Reversal is allowed only when the requested wheel target reverses.
        if target > 0.0:
            output_lower, output_upper = 0.0, self.output_limit
        elif target < 0.0:
            output_lower, output_upper = -self.output_limit, 0.0
        else:
            output_lower, output_upper = 0.0, 0.0
        output = clamp(unconstrained, output_lower, output_upper)

        # Do not integrate farther into output saturation.
        saturated_high = unconstrained > output_upper and error > 0.0
        saturated_low = unconstrained < output_lower and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral
        self.previous_error = error
        return output


class MotorControlPID(Node):
    """Regulate PiBot's left and right wheel velocities from cmd_vel."""

    def __init__(self) -> None:
        super().__init__("motor_control_pid")
        self._declare_parameters()

        self._wheel_radius = float(self.get_parameter("wheel_radius").value)
        self._track_width = float(self.get_parameter("track_width").value)
        self._control_frequency = float(
            self.get_parameter("control_frequency").value
        )
        self._command_timeout = float(self.get_parameter("command_timeout").value)
        self._encoder_timeout = float(self.get_parameter("encoder_timeout").value)
        self._max_linear_acceleration = float(
            self.get_parameter("max_linear_acceleration").value
        )
        self._max_angular_acceleration = float(
            self.get_parameter("max_angular_acceleration").value
        )
        self._filter_alpha = float(
            self.get_parameter("velocity_filter_alpha").value
        )
        self._max_wheel_speed = float(
            self.get_parameter("max_wheel_speed").value
        )
        self._zero_speed_epsilon = float(
            self.get_parameter("zero_speed_epsilon").value
        )
        self._left_motor_direction = float(
            self.get_parameter("left_motor_direction").value
        )
        self._right_motor_direction = float(
            self.get_parameter("right_motor_direction").value
        )
        self._left_encoder_direction = float(
            self.get_parameter("left_encoder_direction").value
        )
        self._right_encoder_direction = float(
            self.get_parameter("right_encoder_direction").value
        )
        self._validate_parameters()

        common_pid_arguments = {
            "integral_limit": float(self.get_parameter("integral_limit").value),
            "output_limit": float(self.get_parameter("max_motor_command").value),
        }
        self._left_pid = PIDController(
            kp=float(self.get_parameter("left_kp").value),
            ki=float(self.get_parameter("left_ki").value),
            kd=float(self.get_parameter("left_kd").value),
            feedforward=float(self.get_parameter("left_feedforward").value),
            **common_pid_arguments,
        )
        self._right_pid = PIDController(
            kp=float(self.get_parameter("right_kp").value),
            ki=float(self.get_parameter("right_ki").value),
            kd=float(self.get_parameter("right_kd").value),
            feedforward=float(self.get_parameter("right_feedforward").value),
            **common_pid_arguments,
        )

        # Pin assignments and direction signs preserve the behavior of the
        # original open-loop motor_control node.
        self._left_motor = Motor(forward=25, backward=18)
        self._right_motor = Motor(forward=14, backward=15)

        self._desired_linear = 0.0
        self._desired_angular = 0.0
        self._command_linear = 0.0
        self._command_angular = 0.0
        self._target_left = 0.0
        self._target_right = 0.0
        self._measured_left = 0.0
        self._measured_right = 0.0
        self._previous_left_degrees: float | None = None
        self._previous_right_degrees: float | None = None
        self._previous_encoder_time: float | None = None
        self._last_encoder_time: float | None = None
        self._last_command_time: float | None = None
        self._encoder_ready = False
        self._reported_wait_reason = ""
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)

        self.create_subscription(Twist, "/cmd_vel", self._command_callback, 10)
        self.create_subscription(
            JointState, "/joint_states", self._encoder_callback, 10
        )
        self._state_publisher = self.create_publisher(
            Float64MultiArray, "~/state", 10
        )
        self._odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), 10
        )
        self.create_timer(1.0 / self._control_frequency, self._control_callback)

        self.get_logger().info(
            "PID motor controller started: wheel_radius=%.3f m, "
            "track_width=%.3f m" % (self._wheel_radius, self._track_width)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("wheel_radius", 0.079375)
        self.declare_parameter("track_width", 0.130175)
        self.declare_parameter("control_frequency", 20.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("encoder_timeout", 0.25)
        self.declare_parameter("max_linear_acceleration", 0.25)
        self.declare_parameter("max_angular_acceleration", 0.5)
        self.declare_parameter("velocity_filter_alpha", 0.25)
        self.declare_parameter("max_wheel_speed", 20.0)
        self.declare_parameter("zero_speed_epsilon", 0.02)

        # Output is the normalized gpiozero Motor command in [-1, 1]. These
        # are conservative starting gains and must be tuned on the robot.
        self.declare_parameter("left_kp", 0.015)
        self.declare_parameter("left_ki", 0.04)
        self.declare_parameter("left_kd", 0.0)
        self.declare_parameter("left_feedforward", 0.05)
        self.declare_parameter("right_kp", 0.015)
        self.declare_parameter("right_ki", 0.04)
        self.declare_parameter("right_kd", 0.0)
        self.declare_parameter("right_feedforward", 0.05)
        self.declare_parameter("integral_limit", 8.0)
        self.declare_parameter("max_motor_command", 1.0)

        self.declare_parameter("left_motor_direction", -1.0)
        self.declare_parameter("right_motor_direction", 1.0)
        self.declare_parameter("left_encoder_direction", -1.0)
        self.declare_parameter("right_encoder_direction", 1.0)
        self.declare_parameter("odom_topic", "/odom_encoder")
        self.declare_parameter("odom_frame", "odom_encoder")
        self.declare_parameter("base_frame", "base_link")

    def _validate_parameters(self) -> None:
        positive = {
            "wheel_radius": self._wheel_radius,
            "track_width": self._track_width,
            "control_frequency": self._control_frequency,
            "command_timeout": self._command_timeout,
            "encoder_timeout": self._encoder_timeout,
            "max_linear_acceleration": self._max_linear_acceleration,
            "max_angular_acceleration": self._max_angular_acceleration,
            "max_wheel_speed": self._max_wheel_speed,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self._filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        if self._zero_speed_epsilon < 0.0:
            raise ValueError("zero_speed_epsilon must be nonnegative")
        integral_limit = float(self.get_parameter("integral_limit").value)
        max_motor_command = float(self.get_parameter("max_motor_command").value)
        if not math.isfinite(integral_limit) or integral_limit <= 0.0:
            raise ValueError("integral_limit must be positive")
        if not 0.0 < max_motor_command <= 1.0:
            raise ValueError("max_motor_command must be in (0, 1]")
        for name in (
            "left_kp",
            "left_ki",
            "left_kd",
            "left_feedforward",
            "right_kp",
            "right_ki",
            "right_kd",
            "right_feedforward",
        ):
            if not math.isfinite(float(self.get_parameter(name).value)):
                raise ValueError(f"{name} must be finite")
        for name, direction in (
            ("left_motor_direction", self._left_motor_direction),
            ("right_motor_direction", self._right_motor_direction),
            ("left_encoder_direction", self._left_encoder_direction),
            ("right_encoder_direction", self._right_encoder_direction),
        ):
            if direction not in (-1.0, 1.0):
                raise ValueError(f"{name} must be either -1.0 or 1.0")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _command_callback(self, message: Twist) -> None:
        linear = float(message.linear.x)
        angular = float(message.angular.z)
        if not math.isfinite(linear) or not math.isfinite(angular):
            self.get_logger().warning("Ignoring non-finite cmd_vel")
            return

        self._desired_linear = linear
        self._desired_angular = angular
        self._last_command_time = self._now()

    def _encoder_callback(self, message: JointState) -> None:
        try:
            if "left" in message.name and "right" in message.name:
                left_index = message.name.index("left")
                right_index = message.name.index("right")
            else:
                left_index, right_index = 0, 1
            left_degrees = float(message.position[left_index])
            right_degrees = float(message.position[right_index])
        except (IndexError, TypeError, ValueError):
            self.get_logger().warning(
                "Ignoring joint_states without left/right encoder positions"
            )
            return
        if not math.isfinite(left_degrees) or not math.isfinite(right_degrees):
            self.get_logger().warning("Ignoring non-finite encoder positions")
            return

        now = self._now()
        if (
            self._previous_encoder_time is not None
            and self._previous_left_degrees is not None
            and self._previous_right_degrees is not None
        ):
            dt = now - self._previous_encoder_time
            if 1.0e-4 < dt < 1.0:
                left_delta = wrapped_degree_delta(
                    left_degrees, self._previous_left_degrees
                )
                right_delta = wrapped_degree_delta(
                    right_degrees, self._previous_right_degrees
                )
                raw_left = (
                    self._left_encoder_direction * math.radians(left_delta) / dt
                )
                raw_right = (
                    self._right_encoder_direction * math.radians(right_delta) / dt
                )
                alpha = self._filter_alpha
                if not self._encoder_ready:
                    self._measured_left = raw_left
                    self._measured_right = raw_right
                else:
                    self._measured_left += alpha * (
                        raw_left - self._measured_left
                    )
                    self._measured_right += alpha * (
                        raw_right - self._measured_right
                    )
                self._encoder_ready = True
                self._update_encoder_odometry(
                    self._left_encoder_direction * math.radians(left_delta),
                    self._right_encoder_direction * math.radians(right_delta),
                    dt,
                )

        self._previous_left_degrees = left_degrees
        self._previous_right_degrees = right_degrees
        self._previous_encoder_time = now
        self._last_encoder_time = now

    def _update_encoder_odometry(
        self, left_delta_rad: float, right_delta_rad: float, dt: float
    ) -> None:
        left_distance = self._wheel_radius * left_delta_rad
        right_distance = self._wheel_radius * right_delta_rad
        distance = 0.5 * (left_distance + right_distance)
        yaw_delta = (right_distance - left_distance) / self._track_width
        midpoint_yaw = self._odom_yaw + 0.5 * yaw_delta
        self._odom_x += distance * math.cos(midpoint_yaw)
        self._odom_y += distance * math.sin(midpoint_yaw)
        self._odom_yaw = math.atan2(
            math.sin(self._odom_yaw + yaw_delta),
            math.cos(self._odom_yaw + yaw_delta),
        )

        linear_velocity = distance / dt
        angular_velocity = yaw_delta / dt
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._odom_frame
        message.child_frame_id = self._base_frame
        message.pose.pose.position.x = self._odom_x
        message.pose.pose.position.y = self._odom_y
        message.pose.pose.orientation.z = math.sin(0.5 * self._odom_yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * self._odom_yaw)
        message.twist.twist.linear.x = linear_velocity
        message.twist.twist.angular.z = angular_velocity
        self._odom_publisher.publish(message)

    def _control_callback(self) -> None:
        now = self._now()
        if self._last_command_time is None:
            self._stop("waiting for cmd_vel")
            return
        if now - self._last_command_time > self._command_timeout:
            self._desired_linear = 0.0
            self._desired_angular = 0.0
            self._command_linear = 0.0
            self._command_angular = 0.0
            self._target_left = 0.0
            self._target_right = 0.0
            self._stop("cmd_vel timed out")
            return
        if not self._encoder_ready or self._last_encoder_time is None:
            self._stop("waiting for encoder data")
            return
        if now - self._last_encoder_time > self._encoder_timeout:
            self._stop("encoder data timed out")
            return

        dt = 1.0 / self._control_frequency
        self._command_linear = move_toward(
            self._command_linear,
            self._desired_linear,
            self._max_linear_acceleration * dt,
        )
        self._command_angular = move_toward(
            self._command_angular,
            self._desired_angular,
            self._max_angular_acceleration * dt,
        )
        half_track = 0.5 * self._track_width
        self._target_left = clamp(
            (
                self._command_linear
                - half_track * self._command_angular
            ) / self._wheel_radius,
            -self._max_wheel_speed,
            self._max_wheel_speed,
        )
        self._target_right = clamp(
            (
                self._command_linear
                + half_track * self._command_angular
            ) / self._wheel_radius,
            -self._max_wheel_speed,
            self._max_wheel_speed,
        )

        if (
            abs(self._target_left) < self._zero_speed_epsilon
            and abs(self._target_right) < self._zero_speed_epsilon
        ):
            self._stop("")
            return

        left_output = self._left_pid.update(
            self._target_left, self._measured_left, dt
        )
        right_output = self._right_pid.update(
            self._target_right, self._measured_right, dt
        )
        self._drive_motor(
            self._left_motor, self._left_motor_direction * left_output
        )
        self._drive_motor(
            self._right_motor, self._right_motor_direction * right_output
        )
        self._publish_state(left_output, right_output)
        self._reported_wait_reason = ""

    def _publish_state(self, left_output: float, right_output: float) -> None:
        message = Float64MultiArray()
        message.layout.dim = [
            MultiArrayDimension(
                label=(
                    "target_left_rad_s,target_right_rad_s,"
                    "measured_left_rad_s,measured_right_rad_s,"
                    "pid_left,pid_right,applied_left,applied_right,"
                    "encoder_linear_m_s,encoder_angular_rad_s"
                ),
                size=10,
                stride=10,
            )
        ]
        message.data = [
            self._target_left,
            self._target_right,
            self._measured_left,
            self._measured_right,
            left_output,
            right_output,
            self._left_motor_direction * left_output,
            self._right_motor_direction * right_output,
            0.5 * self._wheel_radius * (
                self._measured_left + self._measured_right
            ),
            self._wheel_radius * (
                self._measured_right - self._measured_left
            ) / self._track_width,
        ]
        self._state_publisher.publish(message)

    def _stop(self, reason: str) -> None:
        self._left_motor.stop()
        self._right_motor.stop()
        self._left_pid.reset()
        self._right_pid.reset()
        if reason and reason != self._reported_wait_reason:
            self.get_logger().warning(reason)
        self._reported_wait_reason = reason

    @staticmethod
    def _drive_motor(motor: Motor, command: float) -> None:
        command = clamp(command, -1.0, 1.0)
        if command > 0.0:
            motor.forward(command)
        elif command < 0.0:
            motor.backward(-command)
        else:
            motor.stop()

    def destroy_node(self) -> bool:
        if hasattr(self, "_left_motor"):
            self._left_motor.stop()
            self._right_motor.stop()
            self._left_motor.close()
            self._right_motor.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MotorControlPID()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
