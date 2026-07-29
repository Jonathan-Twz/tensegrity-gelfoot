# Arm Control Quick Start

This folder contains helper launches and scripts for running the MED arm in arm-only mode (no gripper, no gamma) and projecting wrench data to `grasp_frame`.

## Launch sequence

Open terminals and run the following commands as needed:

1) Load arm-only robot description and MoveIt context

- Bridge (URDF + TF + basic nodes)

```
roslaunch med_hardware_interface med_lcm_bridge.launch finger:=none gamma_on_hand:=false gripper_on:=false gamma_attachment_on:=false wsg50:=false
```

- MoveIt (use the arm-only config and Medusa scene)

```
roslaunch med_hardware_interface med.launch finger:=none gamma_on_hand:=false gamma_attachment_on:=false moveit_config_path:=$(rospack find med_moveit_arm_only_config) med_scene:=medusa
```

2) Start FT sensor and project wrench to grasp_frame

- NetFT driver (adjust IP if needed)

```
roslaunch netft_rdt_driver netft.launch ip:=192.168.1.33
```

- Wrench projector (publishes /wrench_grasp_frame)

```
python /home/xili/ws/gelfoot_ws/src/tensegrity-gelfoot/arm_control/helpers/wrench_to_grasp_frame.py
```

3) Publish static TF from med_base to link_ft (for visualization/alignment)

```
roslaunch /home/xili/ws/gelfoot_ws/src/tensegrity-gelfoot/arm_control/launch/static_link_ft_tf.launch
```

Notes:
- Ensure you have sourced your workspace: `source ~/ws/gelfoot_ws/devel/setup.bash`.
- RViz: set Fixed Frame to `med_base` and add a RobotModel display to see the arm and the visualization sphere at `grasp_frame`.

