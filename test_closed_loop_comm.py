#!/usr/bin/env python
"""
Verification script for DREMA Closed-Loop gRPC Communication & Dynamic Suite.
Tests Ping, Frame Streaming, Control Query, and Episode Reset.
"""

import os
import sys
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drema.communication.grpc_client import DremaGrpcClient
from run_drema_dynamic_suite import DremaDynamicSuite


def run_test():
    test_port = 50055
    print(f"--- Starting DREMA Dynamic Suite on port {test_port} ---")
    suite = DremaDynamicSuite(
        port=test_port,
        visualize_pybullet=False,
        device="cpu"
    )
    suite.start()

    # Give server a moment to bind port
    time.sleep(1.0)

    try:
        print("\n--- Testing gRPC Client Connection ---")
        client = DremaGrpcClient(target_address=f"localhost:{test_port}")

        # 1. Test Ping
        alive = client.ping(timeout=3.0)
        assert alive, "Ping failed: Server did not respond."
        print("✓ Step 1: Ping passed successfully!")

        # 2. Test Initial 360 Scan Ingestion (Single-Batch with Semantic Labels)
        print("\n--- Testing Initial 360° Scan Ingestion (Single Batch) ---")
        scan_list = [
            (
                'orbit_0_0',
                (np.random.rand(128, 128, 3) * 255).astype(np.uint8),
                np.ones((128, 128), dtype=np.float32) * 0.75,
                np.eye(4, dtype=np.float32),
                np.eye(3, dtype=np.float32)
            )
        ]
        labels = {'diningTable_visible': 1, 'dynamic_tunnel': 2, 'target': 3}
        res_scan = client.push_initial_scan_batch(scan_list, semantic_labels=labels)
        assert res_scan is not None and res_scan.initial_scan_ready, "Expected initial_scan_ready=True"
        print("✓ Step 2: Initial 360° Scan (Single Batch) processed and acknowledged by server.")

        # 3. Test Real-Time Multi-Camera Streaming
        print("\n--- Testing Real-Time Multi-Camera Streaming ---")
        cam_dict = {
            'front': {
                'rgb': (np.random.rand(128, 128, 3) * 255).astype(np.uint8),
                'depth': np.ones((128, 128), dtype=np.float32) * 0.75,
                'extrinsics': np.eye(4, dtype=np.float32),
                'intrinsics': np.eye(3, dtype=np.float32)
            }
        }
        client.start_streaming()
        for t in range(1, 4):
            client.push_frame_observation(timestep=t, camera_dict=cam_dict)
            time.sleep(0.05)

        print("✓ Step 3: Streamed 3 real-time multi-camera observations to server.")

        # 3. Test Control Query (IDLE mode)
        print("\n--- Testing Control Action (IDLE) ---")
        dummy_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        dummy_dq = [0.0] * 7
        dummy_ee = [0.25, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]

        action_idle = client.request_action(
            timestep=1,
            joint_positions=dummy_q,
            joint_velocities=dummy_dq,
            ee_pose=dummy_ee,
            gripper_open=1.0,
            task_active=False
        )
        assert len(action_idle.joint_velocities) == 7, "Expected 7 joint velocities"
        assert all(v == 0.0 for v in action_idle.joint_velocities), "Expected zero velocities in idle"
        print(f"✓ Step 3: IDLE Action verified: {action_idle.status_message}")

        # 4. Test Control Query (ACTIVE mode with target pose)
        print("\n--- Testing Control Action (ACTIVE with target pose) ---")
        dummy_target = [0.30, 0.10, 0.77, 0.0, 0.0, 0.0, 1.0]
        action_active = client.request_action(
            timestep=2,
            joint_positions=dummy_q,
            joint_velocities=dummy_dq,
            ee_pose=dummy_ee,
            gripper_open=1.0,
            task_active=True,
            target_pose=dummy_target,
            target_available=True
        )
        assert len(action_active.joint_velocities) == 7, "Expected 7 joint velocities"
        assert "0.30" in action_active.status_message, "Expected target coordinates in status message"
        print(f"✓ Step 4: ACTIVE Action verified: vels={action_active.joint_velocities[:3]}... ({action_active.status_message})")

        # 5. Test Episode Reset
        print("\n--- Testing Episode Reset ---")
        reset_ok = client.reset_episode(episode_index=0, task_name="dynamic_drema_test_1")
        assert reset_ok, "Episode reset failed"
        print("✓ Step 5: Reset Episode verified.")

        print("\n=======================================================")
        print("★ ALL CLOSED-LOOP ARCHITECTURE TESTS PASSED SUCCESSFULLY ★")
        print("=======================================================\n")

    finally:
        client.close()
        suite.stop()


if __name__ == "__main__":
    run_test()
