#!/usr/bin/env python
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

model_path = os.path.abspath("assets/franka_panda/panda.xml")
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# 1. Put model and data on MJX device
print("Converting to MJX model and data...")
mjx_m = mjx.put_model(m)
mjx_d = mjx.put_data(m, d)

# 2. Vectorize over a batch of K=100 rollouts
K = 100
H = 15 # Horizon
print(f"Creating batch of K={K} parallel rollouts...")

# Batch of initial states with slight perturbations
q_batch = jnp.tile(mjx_d.qpos, (K, 1))
batch_data = jax.vmap(lambda q: mjx_d.replace(qpos=q))(q_batch)

# Define single step and vectorized step
@jax.jit
def parallel_rollout_step(data):
    return jax.vmap(lambda d: mjx.step(mjx_m, d))(data)

# Warmup JIT compile
print("JIT compiling parallel rollout kernel...")
t0 = time.time()
batch_data = parallel_rollout_step(batch_data)
batch_data.qpos.block_until_ready()
t_compile = time.time() - t0
print(f"✓ JIT compilation finished in {t_compile:.2f}s")

# Benchmark H=15 steps of K=100 trajectories
print(f"Running {H} horizon steps for {K} parallel trajectories ({K * H} total physics steps)...")
t0 = time.time()
for _ in range(H):
    batch_data = parallel_rollout_step(batch_data)
batch_data.qpos.block_until_ready()
t_sim = time.time() - t0

fps = (K * H) / t_sim
print(f"✓ Parallel Rollout completed in {t_sim*1000:.2f} ms ({fps:.0f} physics steps/sec)!")
print(f"✓ Final batch qpos shape: {batch_data.qpos.shape}")
print("\n🎉 MJX PARALLEL ROLLOUT BENCHMARK SUCCESSFUL! 🎉")
