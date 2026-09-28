#!/usr/bin/env python
import os
import sys

print("Importing mujoco...")
import mujoco
print("MuJoCo version:", mujoco.__version__)

print("Importing jax...")
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

print("Importing mjx...")
from mujoco import mjx

model_path = os.path.abspath("assets/franka_panda/panda.xml")
print("Loading model from:", model_path)
m = mujoco.MjModel.from_xml_path(model_path)

print("Transferring to MJX...")
mjx_m = mjx.put_model(m)
print(f"SUCCESS! Loaded Franka Panda into MJX! nq={mjx_m.nq}, nv={mjx_m.nv}")
