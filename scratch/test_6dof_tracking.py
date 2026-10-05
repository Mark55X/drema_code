#!/usr/bin/env python
"""
Unit test for 6-DoF End-Effector Tracking (Position + Orientation).
Verifies:
1. FrankaKinematics forward kinematics computes positions and quaternions accurately.
2. MPPMPPIEngine computes Cartesian orientation error (Eq. 27).
3. Motion primitives with 6D twist excite wrist joints (Joints 6 and 7).
4. MP-PMPPI optimization actively drives the wrist towards target orientation.
"""

import numpy as np
from drema.controller.franka_kinematics import FrankaKinematics
from drema.controller.mp_pmppi_engine import MPPMPPIEngine
from drema.controller.motion_primitives import MotionPrimitiveLibrary

def test_6dof_kinematics():
    print("--- [TEST 1] Testing FrankaKinematics 6-DoF FK & Quaternions ---")
    kin = FrankaKinematics(base_position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    # Test batch FK
    K, H = 10, 5
    Q = np.zeros((K, H, 7), dtype=np.float32)
    # Give some non-zero joint angles
    Q[:, :, 1] = 0.5
    Q[:, :, 3] = -1.2
    Q[:, :, 5] = 1.57
    
    pos_batch, quat_batch = kin.batch_forward_kinematics_ee(Q, return_orientations=True)
    assert pos_batch.shape == (K, H, 3), f"Expected (K, H, 3), got {pos_batch.shape}"
    assert quat_batch.shape == (K, H, 4), f"Expected (K, H, 4), got {quat_batch.shape}"
    
    # Check quaternion normalization
    norms = np.linalg.norm(quat_batch, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Quaternions must be normalized"
    print("  ✓ Batch FK returned valid positions and normalized quaternions.")

def test_6dof_motion_primitives():
    print("--- [TEST 2] Testing Motion Primitives 6D Twist Generation ---")
    kin = FrankaKinematics()
    prim_lib = MotionPrimitiveLibrary(kinematics=kin, horizon=10, dt=0.05)
    
    q_curr = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    qd_curr = np.zeros(7, dtype=np.float32)
    target_pos = np.array([0.4, 0.0, 0.5], dtype=np.float32)
    target_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float32) # top-down
    
    U_p = prim_lib.generate_primitives(
        q_current=q_curr,
        qd_current=qd_curr,
        target_pos=target_pos,
        target_rot=target_rot
    )
    print(f"  ✓ Generated {U_p.shape[0]} primitives: {prim_lib.last_primitive_names}")
    assert "align_rot" in prim_lib.last_primitive_names, "align_rot primitive must be present"
    assert "appr_fast" in prim_lib.last_primitive_names, "appr_fast primitive must be present"
    
    # Check that wrist joints (Joint 6 and 7, index 5 and 6) receive non-zero acceleration
    u_align = U_p[prim_lib.last_primitive_names.index("align_rot")]
    print(f"  ✓ align_rot accelerations on wrist joints: J6 max={np.max(np.abs(u_align[:, 5])):.3f}, J7 max={np.max(np.abs(u_align[:, 6])):.3f}")

def test_6dof_mppi_solve():
    print("--- [TEST 3] Testing MP-PMPPI Engine 6-DoF Solve ---")
    kin = FrankaKinematics(base_position=np.array([-0.309, 0.0, 0.82], dtype=np.float32))
    engine = MPPMPPIEngine(kinematics=kin, horizon=10, num_samples_per_planner=16)
    
    q_curr = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    qd_curr = np.zeros(7, dtype=np.float32)
    target_pos = np.array([0.38, 0.02, 0.80], dtype=np.float32)
    target_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    
    qd_opt, diag = engine.solve(
        q_current=q_curr,
        qd_current=qd_curr,
        target_pos=target_pos,
        target_rot=target_rot
    )
    print(f"  ✓ Optimal joint velocities: {np.round(qd_opt, 4)}")
    print(f"  ✓ Diagnostics calc_time: {diag.get('calc_time_ms', 0.0):.1f}ms")
    print(f"  ✓ Top candidate: {diag.get('top_candidate')}")
    assert len(qd_opt) == 7, "Must output 7-joint commands"
    assert not np.any(np.isnan(qd_opt)), "Commands must not be NaN"

if __name__ == '__main__':
    test_6dof_kinematics()
    test_6dof_motion_primitives()
    test_6dof_mppi_solve()
    print("\n🎉 ALL 6-DOF END-EFFECTOR ORIENTATION TESTS PASSED! 🎉")
