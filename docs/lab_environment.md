# Laboratory environment (real FR3)

`run_lab_client.py` runs the lab robot and cameras as the DREMA environment, in place of CoppeliaSim. The DREMA
server is the same and does not know which environment it talks to. Both clients share one loop,
`drema/client/runner.py`: scan, streaming, control requests, keyframes, object roles and the console. Only the
environment changes: `drema/client/coppelia.py` or `drema/client/lab/`.

## Processes

| Where | Process | Port | What |
|-------|---------|------|------|
| robot PC (ROS2 sourced) | `python -m lab_bridges.ros2_joints` | 5003 | arm + gripper joint states, hand pose (O_T_EE) |
| robot PC (ROS2 sourced) | `python -m lab_bridges.ros2_command --topic <controller topic>` | 5005 | arm reference stream, gripper Move/Grasp |
| PC with the ZED | `python -m lab_bridges.zed_camera [--serial N]` | 5002 | one ZED, RGB + depth + intrinsics |
| (alternative) robot PC | `python -m lab_bridges.ros2_camera --rgb_topic ... --depth_topic ... --camera_info_topic ...` | 5004 | any RGB-D camera with a ROS2 driver |
| GPU node | `python run_drema_dynamic_system.py --config <server config>` | 50051 | DREMA server |
| any | `python run_lab_client.py --server_address <host>:50051 [--actuate] [--keyframes file]` | | the environment |

Run the bridges with `python -m` from the repository root. They need only `msgpack` and `numpy`, plus `rclpy` /
`franka_msgs` (ROS2) or `pyzed` (ZED SDK). The camera and joint bridges speak the protocol of Drema2's
`zed_camera`, `ros2_camera` and `ros2_joints`, so a bridge started for Drema2 also serves this client. One process
per camera. Each camera has its own port and its own entry in `configs/environments/lab.yaml`.

## Configuration (`configs/environments/lab.yaml`)

- **World frame**: the robot base frame. `world.robot_base_position` translates it if the server should keep its
  CoppeliaSim numbers. With the default (base at the origin) start the server with
  `perception.workspace.table_z_prior` set to the table height in the base frame. The default 0.75 is the
  CoppeliaSim table.
- **Cameras**: a list. `mount: fixed` takes a `T_base_camera` calibration in the format of Drema2's
  `configs/extraction/calibrations`. `mount: wrist` takes a `T_link_camera` calibration, where the link is `ee` or
  `flange`. All calibrations use the OpenCV optical frame. `stream_resolution` / `scan_resolution` set the size sent
  to the server; the intrinsics are rescaled with it.
- **Scan**: `scan.frames_per_camera` frames of every camera, with the arm where it stands. The server accepts any
  number of views.
- **Segmentation**: the FR3 is rendered at the measured joints into each camera (`assets/franka_fr3/fr3_mask.urdf`,
  collision geometry) and labelled `robot`. **Object masks are not implemented yet**: the server receives robot
  labels only, so it reconstructs the objects without instance ids and cannot track them.
  `drema/client/lab/segmentation.py:ObjectSegmenter` is the interface to fill (planned: SAM3 prompts on the scan,
  then a video tracker).
- **Physical properties**: `ObjectSegmenter.physics()` returns object id -> `{'mass', 'friction', 'static'}` from
  whatever model estimates them, and the scan sends it to the server (`FrameObservation.object_physics`). Only the
  physics rollouts of the `mujoco_warp` twin use it; unknown values fall back to `digital_twin.imagination`
  defaults. It returns `{}` for now.
- **Robot**: `ee_pose: measured` sends the hand pose the robot reports. The server's profile self-check therefore
  compares it with its own kinematic model, and a wrong TCP stops the control before any motion. Client and server
  use the same robot profile, `configs/robots/fr3_real.yaml` (see below).
- **Keyframes**: `--keyframes file` with `lab.keyframes.file` (see `configs/environments/keyframes_example.yaml`:
  world-frame poses in the measured hand convention). Use `--keyframes policy --policy module:function` for an
  external policy, which receives a `LabObservation`.

## Robot profile (server side)

The server reads the robot from `robot.profile`. A profile in `configs/robots/` holds:
- joint offsets and TCP;
- `joint_limits` (`q_min`, `q_max`, `qd_max`), used by the MPPI, the primitives, the local and global IK
  (GeoFIK), the posture choice, the transit planner and the twins;
- `twin`: the PyBullet URDF, the MuJoCo MJCF and the MuJoCo hand bodies.

`digital_twin.urdf_path` / `mujoco_model_path` override the profile's models only when set.

For the lab, start the server with `robot.profile: configs/robots/fr3_real.yaml`. The FR3 has the Panda's DH chain
and hand. It has its own joint limits: the narrower datasheet set, which holds on every FR3; newer
franka_description releases allow wider ranges. It also has its own link geometry:
- `assets/franka_fr3/fr3_twin.urdf`: link0 as root, arm joints first, as the twin's link mapping requires;
- `assets/franka_fr3/fr3.xml`.

Both are built from Drema2's FR3 models by `scripts/build_fr3_assets.py`. `assets/` is git-ignored: add
`assets/franka_fr3` with `git add -f`.

## Safety

- **Shadow mode by default.** Without `--actuate` nothing is sent to the robot. Streaming, scan and MPC run, and the
  commands are only computed.
- **Client side.** The MPC velocities are scaled to `robot.actuation.max_velocity` and integrated into a joint
  reference. That reference stays within `max_lead` of the measured joints and `limit_margin` inside the FR3 limits.
  On a safety stop, a pause or a lost server, the client sends `hold`.
- **Bridge side.**
  - Nothing is published before the first command.
  - Per-joint speed and acceleration are bounded.
  - Without commands for `--watchdog_s` the arm ramps to a stop.
  - A reference more than `--max_reference_jump` from the published one is rejected.
- **Controller.** The controller behind `--topic` must take the stream at `--publish_hz` (250 Hz default) and
  interpolate it to 1 kHz. libfranka aborts on discontinuous references, and a ros2_control forward command
  controller alone does not interpolate. This part depends on the controllers installed on the robot PC and has not
  been tried on the robot.

## Known gaps

- Object segmentation (above).
- The bridge (`lab_bridges/ros2_command.py`) runs without the repository config, so it keeps its own copy of the FR3
  limits. Keep it in line with `fr3_real.yaml`.
- Each camera streams from its own thread at `--cam_fps`, off the control loop. A frame is posed and labelled with
  the joints read right after its grab and is stamped with that time on the client clock. The camera exposure
  time is not used.
- The server's Gaussian renderer ignores the principal point (fine for CoppeliaSim, a few pixels off on real
  cameras). It only affects the photometric term and the viewer.
