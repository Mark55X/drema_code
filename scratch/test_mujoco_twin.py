#!/usr/bin/env python
import os
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin

# 1. Create a dummy test cube mesh
test_mesh_dir = "scratch/test_assets"
os.makedirs(test_mesh_dir, exist_ok=True)
mesh_path = os.path.join(test_mesh_dir, "test_cube.obj")
box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
box.export(mesh_path)

print("--- [TEST 1] Instantiating MuJoCoDigitalTwin ---")
twin = MuJoCoDigitalTwin(
    visualize=False,
    table_z=0.75,
    tracking_mode="constraint", # Should resolve to 'mocap'
    sim_substeps=2
)
print("✓ Twin instantiated.")

print("\n--- [TEST 2] Loading Robot ---")
success = twin.load_robot(base_position=(0.0, 0.0, 0.75))
assert success, "Robot load failed!"
pos_ee, quat_ee = twin.get_ee_pose()
print(f"✓ Robot loaded! End-effector pose: pos={pos_ee}")

print("\n--- [TEST 3] Robot Joint Sync & Kinematics ---")
q_test = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
twin.sync_robot_state(q_test)
pos_ee2, _ = twin.get_ee_pose()
print(f"✓ Synced joints! New EE position: {pos_ee2}")

print("\n--- [TEST 4] Spawning Table ---")
table_id = twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 0.8, -0.4, 0.4))
assert table_id > 0, "Table spawn failed!"
print(f"✓ Table spawned with ID: {table_id}")

print("\n--- [TEST 5] Spawning Mesh Obstacle ---")
obs_id = twin.spawn_scanned_mesh_obstacle(
    mesh_path=mesh_path,
    initial_pos=(0.4, 0.0, 0.85),
    name="test_cube",
    mass=1.0
)
assert obs_id == 0, f"Expected obstacle ID 0, got {obs_id}"
print(f"✓ Obstacle spawned with ID: {obs_id}")

print("\n--- [TEST 6] Syncing Obstacle Pose & Tracking ---")
twin.sync_object_pose(obs_id, position=(0.45, 0.05, 0.85), orientation=(0, 0, 0, 1))
for _ in range(50):
    twin.step_simulation()

pos_res, quat_res = twin.get_object_pose(obs_id)
err = np.linalg.norm(np.array(pos_res) - np.array([0.45, 0.05, 0.85]))
print(f"✓ Object tracked to: {pos_res} | Error: {err*1000:.2f} mm")
assert err < 0.005, f"Tracking error too high: {err}"

print("\n--- [TEST 7] Distance & Closest Points Querying ---")
min_d = twin.get_min_obstacle_distance()
print(f"✓ Minimum obstacle distance: {min_d:.3f} m")
closest = twin.get_closest_points(distance=1.0)
print(f"✓ Closest contact points found: {len(closest)}")

print("\n--- [TEST 8] Collision Checking ---")
has_coll = twin.check_collision()
print(f"✓ In collision: {has_coll}")

print("\n--- [TEST 9] Reset & Shutdown ---")
twin.reset()
assert len(twin.tracked_objects) == 0, "Tracked objects not cleared!"
twin.shutdown()
print("✓ Reset and shutdown clean!")

print("\n🎉 ALL MUJOCO DIGITAL TWIN TESTS PASSED! 🎉")
