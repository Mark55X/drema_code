# MuJoCo Warp twin (`digital_twin.engine: mujoco_warp`)

`drema/simulation/mujoco_warp_twin.py:MuJoCoWarpDigitalTwin` is the MuJoCo twin with its batched work on
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). It has three jobs:

| Job | Worlds | Runs on |
|-----|--------|---------|
| Present state and per-cycle queries (sync, proximity, closest points, contact check) | 1 | MuJoCo C, unchanged |
| MPPI collision costs (`compute_trajectory_collision_costs`) | one per (rollout, evaluated step) | Warp on a GPU, else MuJoCo C |
| Physics imagination (`imagine`, `imagine_async`) | samples x physical worlds | Warp on a GPU, else `mujoco.rollout` |

Without a CUDA device (`device: auto`), both batched jobs use the C code. The MPPI collision costs are then the
MuJoCo twin's, bit for bit.

## Install

```bash
pip install "mujoco-warp==3.14.0"   # into inference_env; adds mujoco-warp and warp-lang only
```

Keep `mujoco-warp` at the version of `mujoco`: on PyPI its versions follow MuJoCo's. `warp-lang` 1.19 bundles the
CUDA 13 toolkit, so the GPU machine needs an NVIDIA driver that supports CUDA 13 (on WSL2, the Windows driver).
On CPU, Warp compiles its kernels once (a few minutes) into `~/.cache/warp`.

## Configuration (`configs/drema_default.yaml`)

- `digital_twin.warp`: `device`, `collision_backend` (`auto`, `mjwarp`, `mujoco_c`), `contacts_per_world`.
- `digital_twin.imagination`: `backend`, `held_object` (`weld` or `grasp`), `grasp_force`, `timestep`,
  `cpu_threads`, buffer sizes, `default_mass` / `default_friction`, `penetration_exclude_depth`, `randomization`
  (`mass_scale`, `friction_scale`, `position_noise`).

## MPPI collision queries on Warp

Each evaluated configuration is one world: the robot at Q[k, h], the obstacles at their forecast poses, the held
objects at their attached poses. Contacts with a margin act as distance queries, so a contact has the distance
`mj_geomDistance` returns. The records then go through `aggregate_gvm_costs` unchanged. Two details:

- A held object is a mocap body in the twin. In the query model it becomes a free body placed through its qpos.
  Mocap-mocap pairs only report penetrations in MuJoCo Warp, never positive distances.
- Obstacle margins are the largest per-pair margin (roles: FRAGILE, approach), and the records are filtered per pair
  as in the C loop.

Measured on 48 put_groceries configurations (8 x 48 rollouts, with role margins and a held object), CPU device:
the same records as the C loop. The closest distance per object differs by at most 0.005 mm (float32). One finger-pad
box group in 435 differs by 0.25 mm, which changes the GVM cost of that configuration by up to 7.5%.

## Physics imagination

```python
from drema.simulation.imagination import ImaginationRequest

req = ImaginationRequest(arm_targets=Q_targets,     # [S, T, 7] joint targets, measured joint space
                         control_dt=0.05,            # targets interpolated linearly over each control step
                         gripper=None,               # [S, T] open amount, or None
                         worlds_per_sample=8,        # R physical worlds per sample, world 0 nominal
                         record_every=10, seed=0)
result = twin.imagine(req)                 # or: future = twin.imagine_async(req); result = future.result()
result.displacement(), result.tilt_deg()   # [S, R, O]
result.object_pos, result.object_quat      # [S, R, n_rec, O, 3 / 4]
result.valid                               # [S, R]
```

The model is built from the present twin (`ImaginationModel`). It contains the robot MJCF of the profile with its
actuators, the table, and every tracked object as a free body with its V-HACD parts. Objects the environment marks
static stay fixed. The model is cached until objects, held objects or the table change.

- **Held object**: `weld` keeps its current pose in the hand frame. `grasp` leaves it free between the fingers,
  which squeeze with `grasp_force`. The model gripper alone holds only a few newtons.
- **Physical properties**: `object_physics` from the environment (mass, friction, static), else the defaults.
- **Randomization**: world r > 0 scales each object's mass and friction uniformly, plus optional position noise.
- **Pose errors**: object pairs that interpenetrate deeper than `penetration_exclude_depth` at the start do not
  collide with each other (`result.info['excluded_pairs']`). On a put_groceries scene, crackers and strawberry jello
  started 27 mm inside each other, and the contact solver blew the randomized worlds up.
- **Invalid worlds**: MuJoCo resets a world whose accelerations diverge and carries on from qpos0. The C backend
  detects this (the simulation time restarts), and Warp reports buffer overflows. Both mark the world invalid.

Measured on CPU: C and Warp give the same weld and grasp lifts within 0.1 mm. Push outcomes agree within about 3 mm
when the object slides. When it topples, outcomes diverge, as they also do within C alone under 0.01 mm of initial
noise, so compare distributions over worlds. `imagine_async` returns at once and a 20 Hz Python loop keeps its
timing while the rollouts run.

## Physical properties from the environment

`Environment.object_physics` (object id -> `{'mass', 'friction', 'static'}`) travels with the last scan chunk
(`FrameObservation.object_physics`) to `BaseDigitalTwin.set_object_physics`. CoppeliaSim reads them from the
respondable shape of each object (mass, Bullet friction, dynamic flag). In the lab, `ObjectSegmenter.physics()` is
the hook for the model that estimates them. The present-state twins do not use these values.
