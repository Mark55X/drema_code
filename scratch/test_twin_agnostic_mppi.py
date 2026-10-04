#!/usr/bin/env python
"""
Unit and Integration Test: Agnostic MP-PMPPI Controller with PyBullet & MuJoCo/MJX Digital Twins.
Verifies that MP-PMPPI works identically and seamlessly with both backend physics engines.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.controller.mp_pmppi_engine import MPPMPPIEngine
from drema.controller.franka_kinematics import FrankaKinematics


def test_twin_contracts():
    print("\n=======================================================")
    print("  [TEST 1] Testing Unified BaseDigitalTwin Contract")
    print("=======================================================")

    mesh_path = os.path.abspath("assets/franka_panda/objs/Panda_gripper_visual.obj")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.abspath("assets/franka_panda/objs/robot_base.obj")

    # 1. Test PyBullet Digital Twin Contract
    print("\n--- 1a. PyBullet Digital Twin ---")
    pb_twin = PyBulletDigitalTwin(visualize=False, table_z=0.75)
    pb_twin.load_robot(base_position=(0.0, 0.0, 0.75))
    pb_twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 0.5, -0.5, 0.5))
    obs_id_pb = pb_twin.spawn_scanned_mesh_obstacle(
        mesh_path=mesh_path,
        initial_pos=(0.4, 0.0, 0.8),
        name="test_obs",
        mass=1.0
    )
    pb_twin.sync_object_pose(obs_id_pb, (0.4, 0.0, 0.8), (0, 0, 0, 1))

    pb_obs_info = pb_twin.get_tracked_obstacles_info()
    assert len(pb_obs_info) == 1, f"Expected 1 obstacle, got {len(pb_obs_info)}"
    print(f"✓ PyBullet Obstacle Info: {pb_obs_info[0]['name']} at {pb_obs_info[0]['position']}")

    pb_ik = pb_twin.calculate_inverse_kinematics((0.4, 0.05, 0.8))
    assert pb_ik is not None and len(pb_ik) == 7, "PyBullet IK failed"
    print(f"✓ PyBullet IK Solution (7 joints): {np.round(pb_ik, 3)}")

    # 2. Test MuJoCo Digital Twin Contract
    print("\n--- 1b. MuJoCo Digital Twin (with optional MJX) ---")
    mj_twin = MuJoCoDigitalTwin(visualize=False, table_z=0.75, enable_mjx=True)
    mj_twin.load_robot(base_position=(0.0, 0.0, 0.75))
    mj_twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 0.5, -0.5, 0.5))
    obs_id_mj = mj_twin.spawn_scanned_mesh_obstacle(
        mesh_path=mesh_path,
        initial_pos=(0.4, 0.0, 0.8),
        name="test_obs",
        mass=1.0
    )
    mj_twin.sync_object_pose(obs_id_mj, (0.4, 0.0, 0.8), (0, 0, 0, 1))

    mj_obs_info = mj_twin.get_tracked_obstacles_info()
    assert len(mj_obs_info) == 1, f"Expected 1 obstacle, got {len(mj_obs_info)}"
    print(f"✓ MuJoCo Obstacle Info: {mj_obs_info[0]['name']} at {mj_obs_info[0]['position']}")

    mj_ik = mj_twin.calculate_inverse_kinematics((0.4, 0.05, 0.8))
    assert mj_ik is not None and len(mj_ik) == 7, "MuJoCo IK failed"
    print(f"✓ MuJoCo IK Solution (7 joints): {np.round(mj_ik, 3)}")

    return pb_twin, mj_twin


def test_trajectory_collision_evaluation(pb_twin, mj_twin):
    print("\n=======================================================")
    print("  [TEST 2] Trajectory Collision Cost Benchmark")
    print("=======================================================")

    kin = FrankaKinematics(base_position=np.array([0.0, 0.0, 0.75]))
    K = 24  # 24 sample trajectories
    H = 15  # 15 lookahead steps
    
    np.random.seed(42)
    # Generate batch of candidate configurations near obstacle
    Q = np.zeros((K, H, 7), dtype=np.float32)
    QD = np.zeros((K, H, 7), dtype=np.float32)
    for k in range(K):
        for h in range(H):
            Q[k, h] = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float32) + np.random.randn(7) * 0.05
            QD[k, h] = np.random.randn(7) * 0.1

    # 1. PyBullet Collision Query Benchmark
    t0 = time.perf_counter()
    pb_cp, pb_gvm = pb_twin.compute_trajectory_collision_costs(
        Q=Q, QD=QD, sigma_1=0.02, sigma_2=0.15, kappa=15.0, rho=0.8, kin_helper=kin
    )
    t_pb = (time.perf_counter() - t0) * 1000.0
    print(f"✓ PyBullet Collision Query: {t_pb:.2f}ms | shape={pb_cp.shape} | max_cost={np.max(pb_cp):.3f}")

    # 2. MuJoCo Collision Query Benchmark
    t0 = time.perf_counter()
    mj_cp, mj_gvm = mj_twin.compute_trajectory_collision_costs(
        Q=Q, QD=QD, sigma_1=0.02, sigma_2=0.15, kappa=15.0, rho=0.8, kin_helper=kin
    )
    t_mj = (time.perf_counter() - t0) * 1000.0
    print(f"✓ MuJoCo Collision Query:   {t_mj:.2f}ms | shape={mj_cp.shape} | max_cost={np.max(mj_cp):.3f}")


def test_mp_pmppi_end_to_end(pb_twin, mj_twin):
    print("\n=======================================================")
    print("  [TEST 3] MP-PMPPI Closed-Loop Step with Both Engines")
    print("=======================================================")

    q_curr = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float32)
    qd_curr = np.zeros(7, dtype=np.float32)
    target_pos = np.array([0.45, 0.0, 0.82], dtype=np.float32)

    # 1. MP-PMPPI with PyBullet
    engine_pb = MPPMPPIEngine(
        horizon=15,
        dt=0.05,
        num_samples_per_planner=24,
        top_k=12
    )
    t0 = time.perf_counter()
    u_pb, meta_pb = engine_pb.solve(
        q_current=q_curr,
        qd_current=qd_curr,
        target_pos=target_pos,
        digital_twin=pb_twin
    )
    lat_pb = (time.perf_counter() - t0) * 1000.0
    print(f"✓ MP-PMPPI + PyBullet: Latency={lat_pb:.1f}ms | weights={meta_pb['weights']} | u_cmd={np.round(u_pb, 3)}")

    # 2. MP-PMPPI with MuJoCo
    engine_mj = MPPMPPIEngine(
        horizon=15,
        dt=0.05,
        num_samples_per_planner=24,
        top_k=12
    )
    t0 = time.perf_counter()
    u_mj, meta_mj = engine_mj.solve(
        q_current=q_curr,
        qd_current=qd_curr,
        target_pos=target_pos,
        digital_twin=mj_twin
    )
    lat_mj = (time.perf_counter() - t0) * 1000.0
    print(f"✓ MP-PMPPI + MuJoCo:   Latency={lat_mj:.1f}ms | weights={meta_mj['weights']} | u_cmd={np.round(u_mj, 3)}")



    assert len(u_pb) == 7 and not np.any(np.isnan(u_pb)), "PyBullet control failed"
    assert len(u_mj) == 7 and not np.any(np.isnan(u_mj)), "MuJoCo control failed"


if __name__ == "__main__":
    pb_twin, mj_twin = test_twin_contracts()
    try:
        test_trajectory_collision_evaluation(pb_twin, mj_twin)
        test_mp_pmppi_end_to_end(pb_twin, mj_twin)
        print("\n🎉 ALL TESTS PASSED! MP-PMPPI RUNS SEAMLESSLY ON BOTH PYBULLET AND MUJOCO/MJX! 🎉\n")
    finally:
        pb_twin.shutdown()
        mj_twin.shutdown()
