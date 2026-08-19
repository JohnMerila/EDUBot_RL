To get RVIZ working
locally: xhost +local:docker

build work space
colcon build --symlink-install
*when first building the workspace it will take some time to build the lidar package
*it will fail on the first build, re run command and it will work

To source the local ROS workspace 
source install/setup.bash

Start up motor control and wheel encoders
ros2 launch PiBot PiBot_launch.py 

PiBot_launch.py starts the closed-loop motor_control_pid node. It converts
/cmd_vel into wheel angular-velocity targets using the wheel radius and track
width, then uses the degree-valued positions on /joint_states as feedback.
PID gains and motor/encoder direction signs are configured in
PiBot/config/motor_pid.yaml. Tune with the wheels raised off the ground first.
The max_linear_acceleration and max_angular_acceleration parameters slew-limit
all /cmd_vel inputs, including direct commands from teleop_twist_keyboard.
PID telemetry is published on /motor_control_pid/state in this order: target
left/right rad/s, measured left/right rad/s, logical PID left/right, applied
motor left/right, encoder linear m/s, and encoder angular rad/s. Encoder-
integrated pose and velocity are also published on /odom_encoder. A PID output
will coast at zero rather than reverse a wheel to brake overspeed unless the
requested wheel target also reverses.

Start up lidar with visualization
ros2 launch sllidar_ros2 view_sllidar_a1_launch.py

if it doesn't work restart the docker container

For teleop control
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
  -p repeat_rate:=20.0 -p key_timeout:=0.6

The repeat rate keeps cmd_vel fresher than motor_control_pid's 0.5 second
command timeout. The key timeout still publishes a zero command when keyboard
input stops.


Topics for robot:

/

Notes:

If you are adding additional packages via sudo apt install . . . 

Put at the end of Dockerfile with run preceding
* add -y if it requires yes when installing 
