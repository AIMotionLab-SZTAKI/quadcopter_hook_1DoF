# Payload grasping and transportation by a quadrotor with a hook-based manipulator

## Installation
To install this repository, first open a terminal and run the following commands:
```
$ git clone https://github.com/AIMotionLab-SZTAKI/quadcopter_hook_1DoF
$ cd quadcopter_hook_1DoF/
```
Now create a virtual environment:
```
$ python3 -m venv venv
```
Activate it:
```
$ source venv/bin/activate
```
Install the necessary packages
```
$ pip install -e .
```
To run the trajectory optimizations, a valid Mosek licence is needed.
To test the installation run a script, e.g.
```
$ cd quadcopter_hook_onedof/test/
$ python3 test_transportation.py
```