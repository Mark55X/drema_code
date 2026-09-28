#!/usr/bin/env python
import os
import mujoco

assets_dir = os.path.abspath("assets/franka_panda/objs")

# Simple MJCF test
mjcf_xml = f"""
<mujoco model="franka_panda">
  <compiler angle="radian" meshdir="{assets_dir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <mesh name="base" file="robot_base.obj"/>
    <mesh name="link1" file="Panda_link1_respondable.obj"/>
    <mesh name="link2" file="Panda_link2_respondable.obj"/>
    <mesh name="link3" file="Panda_link3_respondable.obj"/>
    <mesh name="link4" file="Panda_link4_respondable.obj"/>
    <mesh name="link5" file="Panda_link5_respondable.obj"/>
    <mesh name="link6" file="Panda_link6_respondable.obj"/>
    <mesh name="link7" file="Panda_link7_respondable.obj"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    
    <body name="robot_base" pos="0 0 0.75">
      <geom type="mesh" mesh="base" rgba="0.9 0.9 0.9 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="mesh" mesh="link1" rgba="0.9 0.9 0.9 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print(f"SUCCESS: MuJoCo model compiled! nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
except Exception as e:
    print(f"FAILED: {e}")
