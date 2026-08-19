# PiBot RF2O integration notes

This directory was cloned from the upstream `ros2` branch of
`MAPIRlab/rf2o_laser_odometry` at commit `b38c68e`.

Local compatibility and safety changes:

- Removed the unused Boost and `cmake_modules` dependencies.
- Removed `rclcpp::Node` inheritance from the algorithm class, which created a
  duplicate hidden ROS node on ROS 2.
- Replaced the removed node clock with `std::chrono::steady_clock` for profiling.
- Wait for the `base_link` to LiDAR transform before initializing.
- Initialize the starting orientation to a valid identity quaternion.
- Initialize the new-scan flag, reject scan-size changes, and throttle warnings.
- Move per-scan timing and pose logs from INFO to DEBUG and avoid false
  "waiting" warnings when the processing loop runs faster than the LiDAR.
- Declare the missing `nav_msgs` build/runtime dependency.

The package was compile-tested on ARM64 ROS 2 Jazzy. Use the PiBot integration
launch rather than the upstream launch file:

```bash
ros2 launch rl_controller rf2o_controller.launch.py
```
