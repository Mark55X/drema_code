#!/usr/bin/env python
"""
Comprehensive Integration & Sanity Test for DREMA Refactored Architecture.
Tests BaseDigitalTwin, BasePerceptionModule, VGMappingPerceptionModule,
scene caching (save & instant restore), and DremaDynamicSuite gRPC orchestration.
"""

import os
import sys
import time
import shutil
import numpy as np
import torch
import trimesh

drema_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thesis_root = os.path.dirname(drema_root)
for p in [drema_root, thesis_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from drema.config import load_config
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.perception.vg_mapping_perception import VGMappingPerceptionModule
from drema.perception.base_perception import InitialScanResult, StreamingUpdateResult
from run_drema_dynamic_system import DremaDynamicSystem
from drema.communication.grpc_client import DremaGrpcClient
from drema.communication.proto import drema_comm_pb2


def create_synthetic_cube_mesh(filepath: str, size: float = 0.08):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    box = trimesh.creation.box(extents=[size, size, size])
    box.export(filepath)
    return filepath


def test_digital_twin():
    print("\n--- [TEST 1] Digital Twin (PyBullet) Contract & Execution ---")
    twin = PyBulletDigitalTwin(visualize=False, table_z=0.75)
    
    # 1. Load robot on table surface
    success = twin.load_robot(base_position=(0.0, 0.0, 0.75), joint_positions=[0.0]*7)
    assert success is True and twin.robot_id >= 0, "Robot failed to load"
    print("  ✓ Robot loaded with ID:", twin.robot_id)
    
    # 2. Table
    table_id = twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 0.8, -0.4, 0.4))
    assert table_id >= 0, "Table failed to spawn"
    print("  ✓ Table spawned with ID:", table_id)
    
    # 3. Obstacle with explicit obj_id
    mesh_path = create_synthetic_cube_mesh("assets/scanned_meshes/test_cube.obj")
    obs_body_id = twin.spawn_scanned_mesh_obstacle(
        mesh_path=mesh_path,
        initial_pos=(0.4, 0.0, 0.80),
        obj_id=0,
        name="test_cube"
    )
    assert obs_body_id >= 0, "Obstacle failed to spawn"
    print("  ✓ Obstacle spawned with body ID:", obs_body_id)
    
    # 4. Sync pose (dynamic tracking constraint converges over physics steps)
    twin.sync_object_pose(obj_id=0, position=(0.45, 0.05, 0.80), orientation=(0, 0, 0, 1))
    for _ in range(25):
        twin.step()
    pos, quat = twin.get_object_pose(obj_id=0)
    assert abs(pos[0] - 0.45) < 1e-2 and abs(pos[1] - 0.05) < 1e-2, f"Unexpected pos: {pos}"
    print("  ✓ Object pose successfully synchronized to:", pos)
    
    # 5. Step simulation
    twin.step()
    print("  ✓ Physics step simulation successful")
    
    # 6. Reset & shutdown
    twin.reset()
    twin.shutdown()
    print("  ✓ Digital Twin reset and shutdown cleanly")


def test_mujoco_digital_twin():
    print("\n--- [TEST 1b] Digital Twin (MuJoCo) Contract & Execution ---")
    twin = MuJoCoDigitalTwin(visualize=False, table_z=0.75, tracking_mode="constraint")

    # 1. Load robot
    success = twin.load_robot(base_position=(0.0, 0.0, 0.75), joint_positions=[0.0]*7)
    assert success is True and twin.robot_id >= 0, "MuJoCo Robot failed to load"
    print("  ✓ MuJoCo Robot loaded with ID:", twin.robot_id)

    # 2. Table
    table_id = twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 0.8, -0.4, 0.4))
    assert table_id >= 0, "MuJoCo Table failed to spawn"
    print("  ✓ MuJoCo Table spawned with ID:", table_id)

    # 3. Obstacle
    mesh_path = create_synthetic_cube_mesh("assets/scanned_meshes/test_cube_mj.obj")
    obs_body_id = twin.spawn_scanned_mesh_obstacle(
        mesh_path=mesh_path,
        initial_pos=(0.4, 0.0, 0.80),
        obj_id=0,
        name="test_cube_mj"
    )
    assert obs_body_id >= 0, "MuJoCo Obstacle failed to spawn"
    print("  ✓ MuJoCo Obstacle spawned with ID:", obs_body_id)

    # 4. Sync pose
    twin.sync_object_pose(obj_id=0, position=(0.45, 0.05, 0.80), orientation=(0, 0, 0, 1))
    for _ in range(25):
        twin.step()
    pos, quat = twin.get_object_pose(obj_id=0)
    assert abs(pos[0] - 0.45) < 1e-2 and abs(pos[1] - 0.05) < 1e-2, f"Unexpected pos: {pos}"
    print("  ✓ MuJoCo Object pose synchronized to:", pos)

    # 5. Step & distance
    twin.step()
    min_d = twin.get_min_obstacle_distance()
    assert min_d > 0, "Expected positive distance to obstacle"
    print(f"  ✓ MuJoCo Min distance calculated: {min_d:.3f}m")

    # 6. Reset & shutdown
    twin.reset()
    twin.shutdown()
    print("  ✓ MuJoCo Digital Twin reset and shutdown cleanly")


def test_perception_and_caching():
    print("\n--- [TEST 2] Perception Module & Caching (Save & Instant Restore) ---")
    cfg = load_config("configs/drema_default.yaml")
    cfg.set_nested("system.device", "cpu")
    cfg.set_nested("perception.cache.enabled", False)
    cfg.set_nested("perception.cache.cache_dir", "scratch/test_cache")
    
    perc = VGMappingPerceptionModule(config=cfg)
    
    # Create synthetic tabletop scene with points
    num_pts = 2000
    # Tabletop points around z=0.75m
    tab_x = np.random.uniform(-0.2, 0.6, num_pts)
    tab_y = np.random.uniform(-0.3, 0.3, num_pts)
    tab_z = np.full(num_pts, 0.75) + np.random.normal(0, 0.002, num_pts)
    tab_pts = np.stack([tab_x, tab_y, tab_z], axis=1).astype(np.float32)
    
    # Obstacle points on top of table
    obs_x = np.random.uniform(0.35, 0.45, 400)
    obs_y = np.random.uniform(-0.05, 0.05, 400)
    obs_z = np.random.uniform(0.77, 0.85, 400)
    obs_pts = np.stack([obs_x, obs_y, obs_z], axis=1).astype(np.float32)
    
    all_mock_pts = np.vstack([tab_pts, obs_pts])
    
    # Synthetic frame data
    H, W = 64, 64
    mock_mask = np.zeros((H, W), dtype=np.int32)
    mock_mask[24:40, 24:40] = 20  # Semantic object 'Roof' ID 20
    mock_frame = {
        'name': 'cam_orbit_0',
        'rgb': np.full((H, W, 3), 180, dtype=np.uint8),
        'depth': np.full((H, W), 1.2, dtype=np.float32),
        'intrinsics': np.array([[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]], dtype=np.float32),
        'extrinsics': np.eye(4, dtype=np.float32),
        'near_clip': 0.1,
        'far_clip': 3.0,
        'mask': mock_mask,
        'point_cloud': all_mock_pts
    }
    
    # 1. Process initial scan
    print("  -> Running process_initial_scan...")
    twin = PyBulletDigitalTwin(visualize=False, table_z=0.75)
    res: InitialScanResult = perc.process_initial_scan(
        scan_frames=[mock_frame],
        semantic_labels={'table': 10, 'panda_link0': 1, 'Roof': 20},
        robot_base_pos=np.array([0.0, 0.0, 0.0]),
        digital_twin=twin,
        reachability_radius=0.95
    )
    assert not res.restored_from_cache
    print(f"  ✓ Initial scan processed: table_z={res.table_z:.3f}m, grid={res.grid_dim}")
    
    # 2. Save cache
    cache_dir = "scratch/test_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    save_ok = perc.save_cache(cache_dir)
    assert save_ok, "Failed to save scene cache"
    assert os.path.exists(os.path.join(cache_dir, "metadata.json"))
    assert os.path.exists(os.path.join(cache_dir, "scene_gaussians.pt"))
    assert os.path.exists(os.path.join(cache_dir, "tsdf_map.pt"))
    print("  ✓ Scene cache successfully saved to:", cache_dir)
    
    # 3. Instant restore via load_cache
    perc_restored = VGMappingPerceptionModule(config=cfg)
    twin2 = PyBulletDigitalTwin(visualize=False, table_z=0.75)
    t_start = time.time()
    res_restored: InitialScanResult = perc_restored.load_cache(cache_dir, digital_twin=twin2)
    t_restore = time.time() - t_start
    assert res_restored is not None
    assert res_restored.restored_from_cache
    assert abs(res_restored.table_z - res.table_z) < 1e-4
    print(f"  ✓ Instant cache restoration successful in {t_restore*1000.0:.2f}ms!")
    
    # 4. Check Viser splats & surface voxels
    splats = perc_restored.get_viser_splats_data()
    print("  ✓ Viser splats formatted, centers shape:", splats['centers'].shape if splats else None)
    vox = perc_restored.get_viser_surface_voxels()
    print("  ✓ Viser surface voxels extracted:", vox['points'].shape if vox else "None (acceptable for 1-view test)")
    
    # 5. Dynamic streaming update step
    mock_views = {
        'front': {
            'rgb': np.full((H, W, 3), 180, dtype=np.uint8),
            'depth': np.full((H, W), 1.2, dtype=np.float32),
            'intrinsics': np.array([[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]], dtype=np.float32),
            'extrinsics': np.eye(4, dtype=np.float32),
            'near_clipping': 0.1,
            'far_clipping': 3.0,
            'mask': np.zeros((H, W), dtype=np.int32)
        }
    }
    update_res: StreamingUpdateResult = perc_restored.update_streaming_frame(
        timestep=1,
        camera_views=mock_views,
        robot_state={'base_pos': np.array([0.0, 0.0, 0.0]), 'joints': [0.0]*7},
        digital_twin=twin2
    )
    print(f"  ✓ Streaming update executed in {update_res.latency_ms:.2f}ms, active Gaussians: {update_res.active_gaussians_count}")
    
    # 6. Shutdown
    perc.shutdown()
    perc_restored.shutdown()
    twin.shutdown()
    twin2.shutdown()
    print("  ✓ Perception resources released cleanly")


def test_grpc_suite_lifecycle():
    print("\n--- [TEST 3] DremaDynamicSystem gRPC Server & Client Roundtrip ---")
    cfg = load_config("configs/drema_default.yaml")
    cfg.set_nested("system.device", "cpu")
    cfg.set_nested("system.grpc_port", 50066)
    cfg.set_nested("system.viser.enabled", False)
    cfg.set_nested("digital_twin.gui", False)
    
    system = DremaDynamicSystem(config=cfg)
    system.start()
    time.sleep(0.5)
    
    client = DremaGrpcClient(target_address="localhost:50066")
    try:
        # 1. Ping
        is_alive = client.ping(timeout=1.0)
        assert is_alive, "gRPC Ping failed"
        print("  ✓ gRPC Server Ping: ALIVE")
        
        # 2. Control Action Request
        act = client.request_action(
            timestep=1,
            joint_positions=[0.0]*7,
            joint_velocities=[0.0]*7,
            ee_pose=[0.4, 0.0, 0.8, 0, 1, 0, 0],
            gripper_open=1.0,
            task_active=False,
            robot_base_pos=[0.0, 0.0, 0.0],
            reachability_radius=0.95
        )
        assert act is not None, "Failed to receive ControlAction"
        assert len(act.joint_velocities) == 7, "Action must have 7 joint velocities"
        print("  ✓ ControlAction received with 7 joints, status:", act.status_message)
        
        # 3. Episode Reset
        res_reset = client.reset_episode(episode_index=1, task_name="test_task")
        assert res_reset, "Episode reset failed"
        print("  ✓ Episode reset confirmed by server: SUCCESS")
        
    finally:
        client.close()
        system.stop()
        print("  ✓ DremaDynamicSystem stopped cleanly")


if __name__ == "__main__":
    test_digital_twin()
    test_mujoco_digital_twin()
    test_perception_and_caching()
    test_grpc_suite_lifecycle()
    print("\n=======================================================")
    print("🎉 ALL DREMA REFACTORING INTEGRATION TESTS PASSED! 🎉")
    print("=======================================================\n")
