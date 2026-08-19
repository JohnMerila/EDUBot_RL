# TLDR:

source local on startup

In separate terminals run:
```bash
ros2 launch PiBot PiBot_launch.py
ros2 launch sllidar_ros2 sllidar_a1_launch.py
ros2 launch rl_controller rf2o_controller.launch.py
```

To send a goal point run:
```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w: 1.0}}}"
```

# PiBot RL goal controller

This ROS 2 package deploys the validated Isaac Lab policy in
`2026-08-18_12-18-31_full_seed42/exported/policy.onnx`. It consumes a planar
pose, a 2-D LiDAR scan, and a stamped goal, then publishes the `Twist` messages
already consumed by `PiBot.motor_control`.

## Interfaces

| Direction | Default topic | Type | Purpose |
| --- | --- | --- | --- |
| Subscribe | `/scan` | `sensor_msgs/LaserScan` | Obstacle ranges from `sllidar_ros2` |
| Subscribe | `/odom` | `nav_msgs/Odometry` | Default pose and body velocity |
| Subscribe | `/goal_pose` | `geometry_msgs/PoseStamped` | Goal location |
| Publish | `/cmd_vel` | `geometry_msgs/Twist` | Command consumed by the existing motor node |
| Publish | `/rl_goal_controller/status` | `std_msgs/String` | Controller state/safety reason |
| Publish | `/rl_goal_controller/goal_reached` | `std_msgs/Bool` | Latched goal result |
| Service | `/rl_goal_controller/enable` | `std_srvs/SetBool` | Enable or stop the controller |

The node publishes zero velocity until it has a fresh pose, a fresh scan, and a
goal. It also stops on stale inputs, frame mismatch, invalid inference output,
or a raw LiDAR return closer than `emergency_stop_distance`.

## Build and run

The development-container image installs ONNX Runtime automatically. In an
already-created Ubuntu 24.04 container, install it for the current user with
the explicit PEP 668 override, then build from the workspace root:

```bash
python3 -m pip install --user --break-system-packages onnxruntime
colcon build --packages-select rl_controller PiBot sllidar_ros2
source install/setup.bash
```

`--user` keeps the wheel out of Ubuntu's managed system directories;
`--break-system-packages` only acknowledges the PEP 668 restriction. Do not
run pip with `sudo`.

Start the existing encoder/motor nodes and LiDAR in separate terminals, then
start the default odometry-backed controller:

```bash
ros2 launch PiBot PiBot_launch.py
ros2 launch sllidar_ros2 sllidar_a1_launch.py
ros2 launch rl_controller controller.launch.py
```

The repository also includes a source build of RF2O. To use RF2O for pose and
velocity instead of an existing `/odom` publisher, start the LiDAR first and
then use the integrated launch file:

```bash
ros2 launch sllidar_ros2 sllidar_a1_launch.py
ros2 launch rl_controller rf2o_controller.launch.py
```

That launch publishes the required static `base_link -> laser` transform,
starts RF2O on `/scan`, publishes `/odom_rf2o`, and starts the policy controller
with the RF2O configuration. Its default LiDAR transform `(x=0.08, y=0.0,
z=0.215, yaw=pi)` matches this robot: the A1 laser-frame `+x` axis points toward
its rear cable, and the cable faces the rear of the robot. The controller uses
the same pi `laser_yaw_offset` to rotate `/scan` into the body-frame ray order
used during training. Measure the physical mounting and override the launch
arguments when it differs, for example:

```bash
ros2 launch rl_controller rf2o_controller.launch.py \
  laser_x:=0.10 laser_z:=0.18 laser_yaw:=3.141592653589793
```

If another node already publishes `base_link -> laser`, set
`publish_laser_tf:=false` to avoid conflicting transforms.

Send a goal in the same coordinate frame as `/odom`:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0}, orientation: {w: 1.0}}}"
```

Before allowing the wheels off the ground, confirm that `/scan` angle zero
points toward the A1 cable and therefore toward the robot's rear `-x` axis, and
verify the sign of `/cmd_vel` on the actual drivetrain. Keep
`laser_yaw_offset` equal to the LiDAR frame's measured yaw relative to the base;
it is pi for this rear-cable mounting. Only one node should command `/cmd_vel`;
use a velocity mux if Nav2 or teleoperation is running too.

## Pose-source adapters

The policy is independent of localization. Select an adapter using
`pose_source_type`; a new message adapter can be added to
`rl_controller/pose_sources.py` without altering inference.

Default wheel/filtered odometry:

```bash
ros2 launch rl_controller controller.launch.py
```

RF2O (expects `nav_msgs/Odometry` on `/odom_rf2o`):

```bash
ros2 launch rl_controller controller.launch.py \
  config_file:=$(ros2 pkg prefix rl_controller)/share/rl_controller/config/rf2o.yaml
```

The command above starts only the controller. Prefer the integrated
`rf2o_controller.launch.py` command when RF2O is not already running.

### Building the vendored RF2O package

For an already-created development container, install its declared ROS/system
dependencies and rebuild the affected packages:

```bash
cd ~/ws
source /opt/ros/jazzy/setup.bash
sudo rosdep update
sudo rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to rf2o_laser_odometry rl_controller
source install/setup.bash
```

Confirm each stage before commanding the motors:

```bash
ros2 topic hz /scan
ros2 topic hz /odom_rf2o
ros2 topic echo /odom_rf2o --once
ros2 run tf2_ros tf2_echo odom base_link
```

The RF2O source is pinned to the upstream ROS 2 branch and locally patched to
avoid its duplicate internal ROS node, wait safely for the LiDAR transform,
initialize a valid identity orientation, reject scan-size changes, and throttle
startup warnings.

Nav2/AMCL localization (`/amcl_pose` supplies the map pose and `/odom` supplies
velocity):

```bash
ros2 launch rl_controller controller.launch.py \
  config_file:=$(ros2 pkg prefix rl_controller)/share/rl_controller/config/nav2_amcl.yaml
```

For the Nav2/AMCL configuration, goals must use `frame_id: map`. This node uses
Nav2 localization data; it does not implement the `NavigateToPose` action
server or Nav2's global planner. Goals are direct point goals for the learned
local policy.

Supported `pose_source_type` values are:

- `odometry` (`nav_msgs/Odometry`), including wheel odometry and RF2O
- `pose_with_covariance_stamped` (`geometry_msgs/PoseWithCovarianceStamped`),
  including AMCL
- `pose_stamped` (`geometry_msgs/PoseStamped`), for another localization node

If a pose-only adapter is used without `velocity_topic`, velocity is estimated
from consecutive poses. Set `velocity_topic` to an odometry source when one is
available.

## Policy contract

The controller reproduces the training observation exactly: 72 ranges from
`-pi` through `pi`, goal distance/sine/cosine bearing, normalized forward/yaw
velocity, and the previous two actions. The output is mapped to forward speed
`[0, 0.8] m/s` and yaw rate `[-1.5, 1.5] rad/s`, then limited to the training
accelerations. The scaling parameters in the YAML are therefore part of the
model contract and should only be changed with renewed validation.
