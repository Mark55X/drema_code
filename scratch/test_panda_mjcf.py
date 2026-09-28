#!/usr/bin/env python
import os
import sys
import numpy as np
import mujoco

# Add repo to sys.path
sys.path.insert(0, os.path.abspath("."))
from drema.controller.franka_kinematics import FrankaKinematics

assets_dir = os.path.abspath("assets/franka_panda/objs")

# Complete Franka Panda MJCF matching standard DH
mjcf_xml = f"""
<mujoco model="franka_panda">
  <compiler angle="radian" meshdir="{assets_dir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom contype="1" conaffinity="1" density="1000"/>
  </default>

  <asset>
    <mesh name="base" file="robot_base.obj"/>
    <mesh name="link1" file="Panda_link1_respondable.obj"/>
    <mesh name="link2" file="Panda_link2_respondable.obj"/>
    <mesh name="link3" file="Panda_link3_respondable.obj"/>
    <mesh name="link4" file="Panda_link4_respondable.obj"/>
    <mesh name="link5" file="Panda_link5_respondable.obj"/>
    <mesh name="link6" file="Panda_link6_respondable.obj"/>
    <mesh name="link7" file="Panda_link7_respondable.obj"/>
    <mesh name="gripper" file="Panda_gripper_visual.obj"/>
  </asset>

  <worldbody>
    <body name="robot_base" pos="0 0 0.75">
      <geom type="mesh" mesh="base" rgba="0.95 0.95 0.95 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="mesh" mesh="link1" rgba="0.95 0.95 0.95 1"/>
        <body name="link2" pos="0 0 0" quat="0.7071068 -0.7071068 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom type="mesh" mesh="link2" rgba="0.95 0.95 0.95 1"/>
          <body name="link3" pos="0 -0.316 0" quat="0.7071068 0.7071068 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="mesh" mesh="link3" rgba="0.95 0.95 0.95 1"/>
            <body name="link4" pos="0.0825 0 0" quat="0.7071068 0.7071068 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom type="mesh" mesh="link4" rgba="0.95 0.95 0.95 1"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="0.7071068 -0.7071068 0 0">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="mesh" mesh="link5" rgba="0.95 0.95 0.95 1"/>
                <body name="link6" pos="0 0 0" quat="0.7071068 0.7071068 0 0">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom type="mesh" mesh="link6" rgba="0.95 0.95 0.95 1"/>
                  <body name="link7" pos="0.088 0 0" quat="0.7071068 0.7071068 0 0">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="mesh" mesh="link7" rgba="0.95 0.95 0.95 1"/>
                    <body name="ee" pos="0 0 0.107">
                      <geom type="mesh" mesh="gripper" rgba="0.3 0.3 0.3 1"/>
                      <site name="tcp" pos="0 0 0.1034"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(mjcf_xml)
data = mujoco.MjData(model)

# Test zero position
q_test = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
data.qpos[:7] = q_test
mujoco.mj_forward(model, data)

tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
tcp_pos_mj = data.site_xpos[tcp_site_id]

kin = FrankaKinematics(base_position=np.array([0.0, 0.0, 0.75]))
tcp_pos_kin, tcp_rot_kin = kin.forward_kinematics_ee(q_test)

err = np.linalg.norm(tcp_pos_mj - tcp_pos_kin)
print(f"MuJoCo TCP Pos: {tcp_pos_mj}")
print(f"FrankaKinematics TCP Pos: {tcp_pos_kin}")
print(f"TCP Position Difference: {err * 1000:.2f} mm")
if err < 0.005:
    print("SUCCESS: MuJoCo model matches analytical Franka kinematics within <5mm!")
else:
    print("WARNING: Discrepancy in kinematics.")
