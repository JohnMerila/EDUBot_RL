"""Pure-numpy implementation of the Isaac Lab policy interface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


POLICY_LIDAR_RAYS = 72
POLICY_OBSERVATIONS = 79
POLICY_ACTIONS = 2


@dataclass(frozen=True)
class PolicyLimits:
    """Scaling values used while training the exported policy."""

    lidar_max_range: float = 8.0
    max_linear_speed: float = 0.8
    max_angular_speed: float = 1.5


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a ROS quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        raise ValueError("pose contains a zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def resample_laser_scan(
    ranges: list[float] | tuple[float, ...] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    policy_max_range: float = 8.0,
    laser_yaw_offset: float = math.pi,
    ray_count: int = POLICY_LIDAR_RAYS,
) -> tuple[np.ndarray, float]:
    """Resample a ROS LaserScan to the policy's body-frame ray convention.

    The Isaac Lab environment casts 72 rays from -pi through pi (exclusive).
    The default pi yaw matches this robot's rear-cable A1 mounting. Missing and
    non-finite returns are represented as maximum range. The second return
    value is the closest valid raw return for the safety stop.
    """
    values = np.asarray(ranges, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("LaserScan must contain at least two ranges")
    if not math.isfinite(angle_increment) or abs(angle_increment) < 1.0e-12:
        raise ValueError("LaserScan angle_increment must be non-zero")
    if policy_max_range <= 0.0 or ray_count <= 0:
        raise ValueError("policy scan limits must be positive")

    valid = np.isfinite(values) & (values >= max(0.0, range_min))
    if math.isfinite(range_max) and range_max > 0.0:
        valid &= values <= range_max
    closest = float(np.min(values[valid])) if np.any(valid) else math.inf
    clean = np.where(valid, values, policy_max_range)
    clean = np.clip(clean, 0.0, policy_max_range)

    angles = angle_min + np.arange(values.size, dtype=np.float64) * angle_increment
    if angle_increment < 0.0:
        angles = angles[::-1]
        clean = clean[::-1]

    body_angles = -math.pi + np.arange(ray_count, dtype=np.float64) * (2.0 * math.pi / ray_count)
    target_angles = body_angles - laser_yaw_offset
    span = abs(angle_increment) * values.size
    if span >= 2.0 * math.pi - 1.5 * abs(angle_increment):
        angles = np.concatenate((angles - 2.0 * math.pi, angles, angles + 2.0 * math.pi))
        clean = np.concatenate((clean, clean, clean))

    sampled = np.interp(target_angles, angles, clean, left=policy_max_range, right=policy_max_range)
    return sampled.astype(np.float32), closest


def build_observation(
    scan_ranges: np.ndarray,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    linear_velocity: float,
    angular_velocity: float,
    previous_action: np.ndarray,
    limits: PolicyLimits = PolicyLimits(),
) -> np.ndarray:
    """Build the exact 79-element observation used during training."""
    scan = np.asarray(scan_ranges, dtype=np.float32)
    action = np.asarray(previous_action, dtype=np.float32)
    if scan.shape != (POLICY_LIDAR_RAYS,):
        raise ValueError(f"expected {POLICY_LIDAR_RAYS} scan rays, got {scan.shape}")
    if action.shape != (POLICY_ACTIONS,):
        raise ValueError(f"expected {POLICY_ACTIONS} previous actions, got {action.shape}")

    delta_x = goal_x - robot_x
    delta_y = goal_y - robot_y
    distance = math.hypot(delta_x, delta_y)
    bearing = wrap_angle(math.atan2(delta_y, delta_x) - robot_yaw)
    goal = np.asarray(
        [
            min(distance, limits.lidar_max_range) / limits.lidar_max_range,
            math.sin(bearing),
            math.cos(bearing),
        ],
        dtype=np.float32,
    )
    velocity = np.clip(
        np.asarray(
            [linear_velocity / limits.max_linear_speed, angular_velocity / limits.max_angular_speed],
            dtype=np.float32,
        ),
        -2.0,
        2.0,
    )
    observation = np.concatenate(
        (np.clip(scan, 0.0, limits.lidar_max_range) / limits.lidar_max_range, goal, velocity, action)
    ).astype(np.float32)
    if observation.shape != (POLICY_OBSERVATIONS,):
        raise AssertionError(f"policy observation has unexpected shape {observation.shape}")
    return observation


def action_to_command(action: np.ndarray, limits: PolicyLimits = PolicyLimits()) -> tuple[float, float]:
    """Map the policy's normalized action to its training-time body command."""
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if clipped.shape != (POLICY_ACTIONS,):
        raise ValueError(f"expected {POLICY_ACTIONS} actions, got {clipped.shape}")
    linear = 0.5 * limits.max_linear_speed * (float(clipped[0]) + 1.0)
    angular = limits.max_angular_speed * float(clipped[1])
    return linear, angular
