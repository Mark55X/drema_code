#!/usr/bin/env python
import os
import mujoco
import numpy as np

# Test mocap + weld constraint tracking in MuJoCo
mjcf_xml = """
<mujoco model="test_tracking">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom name="table" type="box" size="0.4 0.4 0.375" pos="0 0 0.375" rgba="0.7 0.7 0.7 1"/>

    <!-- Object 1: Mocap handle + Dynamic rigid body connected with weld -->
    <body name="obs1_mocap" mocap="true" pos="0 0 0.85">
      <geom type="sphere" size="0.01" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    </body>

    <body name="obs1" pos="0 0 0.85">
      <freejoint name="obs1_joint"/>
      <geom name="obs1_geom" type="box" size="0.05 0.05 0.05" mass="1.0" rgba="0.2 0.45 0.85 1"/>
    </body>
  </worldbody>

  <equality>
    <weld name="obs1_weld" body1="obs1_mocap" body2="obs1" solref="0.01 1.0" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(mjcf_xml)
data = mujoco.MjData(model)

# Initial step
mujoco.mj_step(model, data)
obs1_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obs1")
mocap1_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obs1_mocap")]

# Move mocap target to [0.1, 0.2, 0.9]
data.mocap_pos[mocap1_id] = [0.1, 0.2, 0.9]

for _ in range(50):
    mujoco.mj_step(model, data)

sim_pos = data.xpos[obs1_body_id]
err = np.linalg.norm(sim_pos - np.array([0.1, 0.2, 0.9]))
print(f"Tracking Test: Target=[0.1, 0.2, 0.9], SimPos={sim_pos}, Err={err*1000:.2f} mm")
assert err < 0.005, f"Tracking error too high: {err}"
print("✓ MuJoCo Mocap + Weld tracking is rock-solid!")
