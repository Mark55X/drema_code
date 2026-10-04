import pybullet as p
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

twin = PyBulletDigitalTwin(visualize=False, table_z=0.75, tracking_mode="constraint")
twin.load_robot(base_position=(-0.309, 0.0, 0.82))

mesh_path = os.path.abspath("assets/franka_panda/objs/robot_base.obj")
initial_pos = (0.378, 0.237, 0.813)
body_id = twin.spawn_scanned_mesh_obstacle(
    mesh_path=mesh_path,
    initial_pos=initial_pos,
    name="tunnel_obstacle",
    mass=1.0
)

# Simulate 50 frames arriving at 5 Hz (every 0.2s), with 10 action steps per frame (50 Hz control)
poses = [
    (0.378, 0.237, 0.813),
    (0.379, 0.252, 0.813),
    (0.379, 0.233, 0.813),
    (0.380, 0.211, 0.813),
    (0.379, 0.167, 0.813),
    (0.383, 0.134, 0.813),
    (0.379, 0.097, 0.813),
    (0.380, 0.041, 0.813),
    (0.378, 0.010, 0.813),
    (0.379, -0.031, 0.813),
    (0.379, -0.077, 0.813),
    (0.379, -0.116, 0.813),
    (0.378, -0.150, 0.813),
    (0.379, -0.171, 0.813),
    (0.378, -0.198, 0.813),
]

print("--- Testing 'constraint' tracking mode in PyBullet ---")
for i, target in enumerate(poses):
    twin.sync_object_pose(0, target, (0, 0, 0, 1))
    # 10 control steps between frames (sim_substeps=2 each -> 20 simulation steps)
    for _ in range(10):
        twin.step()
    actual_pos, _ = twin.get_object_pose(0)
    err = np.linalg.norm(np.array(actual_pos) - np.array(target))
    print(f"Step {i:02d} | Target Y: {target[1]:+.3f} | Actual Y: {actual_pos[1]:+.3f} | Error: {err:.4f}m")

twin.shutdown()
