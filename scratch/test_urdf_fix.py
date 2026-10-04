import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

# 1. Test with assets/franka_panda/panda.urdf (current config)
twin_broken = PyBulletDigitalTwin(visualize=False, robot_urdf_path="assets/franka_panda/panda.urdf")
twin_broken.load_robot(base_position=(-0.309, 0.0, 0.82))
mesh_path = os.path.abspath("assets/scanned_meshes/tunnel_obstacle_0.obj")
twin_broken.spawn_scanned_mesh_obstacle(mesh_path, (0.378, -0.08, 0.813), name="tunnel")
# End effector position in PyBullet
link_state_broken = p.getLinkState(twin_broken.robot_id, 6)
print("Broken URDF Link 6 (wrist) Z:", link_state_broken[0][2], "-> min_d:", twin_broken.get_min_obstacle_distance())
twin_broken.shutdown()

# 2. Test with standard franka_panda/panda.urdf
twin_fixed = PyBulletDigitalTwin(visualize=False, robot_urdf_path="franka_panda/panda.urdf")
twin_fixed.load_robot(base_position=(-0.309, 0.0, 0.82))
twin_fixed.spawn_scanned_mesh_obstacle(mesh_path, (0.378, -0.08, 0.813), name="tunnel")
# Reach towards target [0.38, 0.02, 0.80]
q_reach = twin_fixed.calculate_inverse_kinematics((0.38, 0.02, 0.80))
twin_fixed.sync_robot_state(q_reach)
twin_fixed.step()
link_state_fixed = p.getLinkState(twin_fixed.robot_id, 6)
print("Standard URDF Link 6 (wrist) Z:", link_state_fixed[0][2], "-> min_d:", twin_fixed.get_min_obstacle_distance())
twin_fixed.shutdown()
