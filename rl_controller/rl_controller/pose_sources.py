"""Pose-message adapters for the policy controller.

Add another adapter to ``POSE_SOURCE_TYPES`` to support a new localization
message without changing policy inference or safety behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from .policy_contract import yaw_from_quaternion


@dataclass(frozen=True)
class PoseMeasurement:
    x: float
    y: float
    yaw: float
    frame_id: str
    linear_velocity: float | None = None
    angular_velocity: float | None = None


class OdometryPoseSource:
    message_type = Odometry

    @staticmethod
    def extract(message: Odometry) -> PoseMeasurement:
        pose = message.pose.pose
        return PoseMeasurement(
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw_from_quaternion(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ),
            frame_id=message.header.frame_id,
            linear_velocity=message.twist.twist.linear.x,
            angular_velocity=message.twist.twist.angular.z,
        )


class PoseWithCovarianceSource:
    message_type = PoseWithCovarianceStamped

    @staticmethod
    def extract(message: PoseWithCovarianceStamped) -> PoseMeasurement:
        pose = message.pose.pose
        return PoseMeasurement(
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw_from_quaternion(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ),
            frame_id=message.header.frame_id,
        )


class PoseStampedSource:
    message_type = PoseStamped

    @staticmethod
    def extract(message: PoseStamped) -> PoseMeasurement:
        pose = message.pose
        return PoseMeasurement(
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw_from_quaternion(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ),
            frame_id=message.header.frame_id,
        )


POSE_SOURCE_TYPES = {
    "odometry": OdometryPoseSource,
    "pose_with_covariance_stamped": PoseWithCovarianceSource,
    "pose_stamped": PoseStampedSource,
}
