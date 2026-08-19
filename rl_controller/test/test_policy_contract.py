import math
import unittest

import numpy as np

from rl_controller.policy_contract import (
    POLICY_OBSERVATIONS,
    action_to_command,
    build_observation,
    resample_laser_scan,
    wrap_angle,
    yaw_from_quaternion,
)


class PolicyContractTest(unittest.TestCase):
    def test_observation_matches_training_order_and_shape(self):
        scan = np.full(72, 4.0, dtype=np.float32)
        observation = build_observation(
            scan,
            robot_x=1.0,
            robot_y=2.0,
            robot_yaw=math.pi / 2.0,
            goal_x=1.0,
            goal_y=4.0,
            linear_velocity=0.4,
            angular_velocity=-0.75,
            previous_action=np.asarray([0.25, -0.5], dtype=np.float32),
        )

        self.assertEqual(observation.shape, (POLICY_OBSERVATIONS,))
        np.testing.assert_allclose(observation[:72], 0.5)
        np.testing.assert_allclose(observation[72:75], [0.25, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(observation[75:], [0.5, -0.5, 0.25, -0.5])

    def test_full_scan_resamples_and_reports_raw_minimum(self):
        ranges = np.full(360, 8.0)
        ranges[180] = 0.42
        sampled, closest = resample_laser_scan(
            ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 360.0,
            range_min=0.1,
            range_max=12.0,
            laser_yaw_offset=0.0,
        )

        self.assertEqual(sampled.shape, (72,))
        self.assertAlmostEqual(float(sampled[36]), 0.42, places=6)
        self.assertAlmostEqual(closest, 0.42, places=6)

    def test_rear_cable_mount_rotates_laser_scan_into_body_frame(self):
        ranges = np.full(360, 8.0)
        ranges[180] = 0.42  # Laser angle zero points rearward on this robot.
        sampled, closest = resample_laser_scan(
            ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 360.0,
            range_min=0.1,
            range_max=12.0,
        )

        self.assertAlmostEqual(float(sampled[0]), 0.42, places=6)
        self.assertAlmostEqual(closest, 0.42, places=6)

    def test_invalid_and_out_of_range_scan_values_become_max_range(self):
        ranges = np.full(72, math.inf)
        ranges[0] = math.nan
        ranges[1] = 0.01
        sampled, closest = resample_laser_scan(
            ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 72.0,
            range_min=0.1,
            range_max=8.0,
            laser_yaw_offset=0.0,
        )

        np.testing.assert_allclose(sampled, 8.0)
        self.assertEqual(closest, math.inf)

    def test_action_mapping_matches_environment(self):
        np.testing.assert_allclose(action_to_command(np.asarray([-1.0, -1.0])), (0.0, -1.5))
        np.testing.assert_allclose(action_to_command(np.asarray([0.0, 0.0])), (0.4, 0.0))
        np.testing.assert_allclose(action_to_command(np.asarray([1.0, 1.0])), (0.8, 1.5))

    def test_angle_and_quaternion_helpers(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(
            yaw_from_quaternion(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            math.pi / 2.0,
        )
        with self.assertRaises(ValueError):
            yaw_from_quaternion(0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
