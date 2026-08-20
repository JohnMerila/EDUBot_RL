# Overview

This is a repo for running RL navigation algorithms on a physical robot.  This works with the hardware configuration of the EDUBot, a differential drive robot with a lidar for navigation and a pi5 onboard computer.

The system is run in a docker container locally and requires a manual startup of the scripts using the PiBot_Launch launch file

Control of the robot is achievable in Rviz on a control computer with the topics subscribed to.  A goal pose can be sent to the robot and it shown in the map and the robot will autonomously navigate to that location.

The rf20 laser odometry repo was used as an odometry source for the robot - a more robust solution than the onboard encoders.

Demo video:

https://youtube.com/shorts/kGsl8ITXeks

## Future Work

The current RL script here has some excessive angular motion, however, it does eventually reach the goal.  On the robot side a contributor to this is the lack of velocity control for the motors where currently
an open loop controller is implemented.  Future work will implement a velocity controller that should reduce the angular motion due to unexpected angular velocity during linear input velocities.

This could also likely be resolved by training the network with domain randomization in commanded vs. actual velocities, wheel diameters, and etc. to produce a more robust algorithm.

