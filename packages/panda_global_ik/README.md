# panda-global-ik

All inverse-kinematics solutions of the Franka Emika Panda for a TCP pose, in a few milliseconds.

The usual IK solvers (PyBullet `calculateInverseKinematics`, damped least squares, mink, ...) are **local**. They start
from the current joint configuration and refine it step by step. They find one solution, the one in the "valley"
they start in. If the pose is only reachable with a different arm posture (elbow on the other side, joint 7 turned the
other way), they report it as unreachable.

This package wraps [GeoFIK](https://github.com/PabloLopezCustodio/GeoFIK), an analytical solver. For a given value of
joint 7 it returns every arm configuration that reaches the pose (up to 8), and it handles the singular cases. The
package samples joint 7 over its whole range and keeps only the solutions that are inside the joint limits. Every
solution is checked with the package's own forward kinematics. So when `solutions()` returns nothing, the pose is
out of reach, up to the joint-7 sampling described below.

## Installation (Linux, Python >= 3.9)

```bash
sudo apt install g++ libeigen3-dev          # or: conda install -c conda-forge eigen
pip install .                                # compiles GeoFIK (a few seconds)
pip install pytest && pytest tests           # optional, about 30 s
```

Use `pytest tests`, not `python -m pytest`. Run from the project folder, the latter imports the source folder, which
has no compiled library, instead of the installed package.

If Eigen is installed elsewhere, set `EIGEN_INCLUDE` to the folder that contains `eigen3/`.

## Usage

```python
import numpy as np
from panda_global_ik import PandaIK

ik = PandaIK()                          # Franka Hand TCP, poses in the robot base frame
target = ik.fk(np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]))     # any 4x4 TCP pose

sols = ik.solutions(target)             # (n, 7) array of all valid configurations
q = ik.best(sols, q_current)            # closest to q_current among those >= 0.2 rad from every limit
q = ik.ik(target, q_current)            # the same in one call; None = out of reach
sols_q7 = ik.solutions(target, q7=0.5)  # only the configurations with joint 7 at 0.5 rad
```

## Conventions (read before using it on the robot)

| Parameter | Meaning | Real Panda |
|---|---|---|
| `tcp_offset` | 4x4 flange -> TCP transform | default = Franka Hand: 0.1034 m along z, -45 deg about z. Change it for other fingers or tools |
| `joint_offsets` | added to the joint values before the kinematic chain | **zeros** (default). Non-zero only for simulator models whose joint zeros differ, e.g. joint 4 of the RLBench Panda = 0.0683 rad |
| `base` | 4x4 world -> robot base | identity (default) if your poses are already in the base frame |

- Joint values and limits are the official Franka ones (`Q_MIN`, `Q_MAX`), in radians.
- `target` is a 4x4 homogeneous matrix of the TCP in the world frame (world = base by default).
- `fk(q)` uses the same conventions, so `ik.solutions(ik.fk(q))` always contains a configuration in the same family as `q`.

## Setting it up on a real Panda

The solver is only as correct as the robot description you give it. Go through these steps once per robot and tool.

1. **TCP.** With the standard Franka Hand nothing is needed. With other fingers or a tool, pass the flange -> TCP
   transform. It is the `F_T_EE` configured in Franka Desk, available in libfranka as `robot_state.F_T_EE`, a
   column-major array of 16 values:
   ```python
   F_T_EE = np.array(robot_state.F_T_EE).reshape(4, 4).T
   ```
2. **Joint offsets.** Leave the default (zeros) on the real robot.
3. **Base.** If your targets are expressed in a camera or world frame, pass the world -> base transform from your
   calibration. If they are already in the robot base frame (`panda_link0`), leave the default.
4. **Check the model against the robot.** Do it before the first motion, with the robot standing still:
   ```python
   ik = PandaIK(tcp_offset=F_T_EE)                       # plus base=... if needed
   q = np.array(robot_state.q)
   O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T  # TCP pose measured by the robot, base frame
   T = ik.base_inv @ ik.fk(q)                             # model TCP pose, base frame
   print("position error [mm]:", 1000 * np.linalg.norm(T[:3, 3] - O_T_EE[:3, 3]))
   ```
   A few tenths of a millimetre to about 2 mm is normal. Errors of centimetres mean a wrong `tcp_offset` (or `base`,
   when you compare in the world frame).

### Common mistakes

- Passing the **flange** pose instead of the **TCP** pose. `target` is the TCP pose, consistent with `tcp_offset`.
- Reading the libfranka 4x4 arrays as row-major: they are column-major (`.reshape(4, 4).T`).
- Mixing units: metres and radians everywhere. Joints are ordered J1..J7.
- Running `python -m pytest` from the project folder (see Installation): it imports the uncompiled sources.

## How to use it well

- Call it **when the posture is decided**: once per target or waypoint. Then let your controller or a local IK track
  the target from the chosen configuration. This keeps the motion continuous and avoids the local IK getting stuck in
  the wrong family.
- `best()` picks the configuration closest to the current one, which means the least arm motion. A closer solution can
  still need a large reconfiguration when the only reachable family is far away. Check `np.linalg.norm(q - q_current)`
  if large motions matter in your application.
- Joint 7 is sampled every 10 deg (`q7_samples=33`). About 0.2 % of the reachable poses are reachable only in a
  narrower joint-7 window. When the first pass finds nothing, a second pass at 2.5 deg (`q7_refine_samples=133`) runs.
  With both passes, 4000 random reachable poses out of 4000 were solved.

## Performance (laptop CPU, Python)

- one GeoFIK call: ~15 us. `solutions()`: ~11 ms on average with the verification of every solution. The denser
  second pass (~42 ms) runs only for the rare poses described above.
- 100 % of 4000 random reachable poses and of 700 keyframes recorded from RLBench tasks solved. Starting from the
  RLBench start posture, a local damped-least-squares IK solved 70 % of the same poses and PyBullet's IK 1 %.
- No wrong solution: unlike the He & Liu solver (panda-py, frantik), GeoFIK never returned a configuration that misses
  the pose. The package checks every solution anyway.

## Limitations

- Franka Emika Panda (FER) joint limits. The FR3 has different limits and has not been tested.
- Self-collisions and collisions with the environment are not checked.

## Credits and licence

- GeoFIK: P. C. Lopez-Custodio, Y. Gong, L. F. C. Figueredo, *GeoFIK: A Fast and Reliable Geometric Solver for the IK of
  the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters*, arXiv:2503.03992. Apache-2.0, see
  `third_party/geofik/LICENSE`. Included unmodified from commit `b22a8fdaed3aca52b440a650bba35d099cbde178`.
- `src/geofik_c.cpp` and `panda_global_ik/` are the wrapper (C entry points, joint-7 sampling, verification, selection).
