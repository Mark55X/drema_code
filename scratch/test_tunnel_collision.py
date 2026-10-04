import pybullet as p
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

twin = PyBulletDigitalTwin(visualize=False, table_z=0.75, tracking_mode="constraint")
twin.load_robot(base_position=(-0.309, 0.0, 0.82))

mesh_path = os.path.abspath("assets/scanned_meshes/tunnel_obstacle_0.obj")
print(f"Mesh path exists: {os.path.exists(mesh_path)}")

obstacle_pos = (0.378, -0.08, 0.813)
body_id = twin.spawn_scanned_mesh_obstacle(
    mesh_path=mesh_path,
    initial_pos=obstacle_pos,
    name="tunnel_obstacle",
    mass=1.0,
    obj_id=84
)

# Target: [0.38, 0.02, 0.80]
q_ik = twin.calculate_inverse_kinematics((0.38, 0.02, 0.80))
twin.sync_robot_state(q_ik)
twin.step()

# Check min_dist
min_d = twin.get_min_obstacle_distance()
print("get_min_obstacle_distance():", min_d)

pts = p.getClosestPoints(bodyA=twin.robot_id, bodyB=body_id, distance=0.5)
print("Closest points count (dist 0.5):", len(pts))
if pts:
    for pt in pts[:5]:
        print(f"  Link {pt[3]} -> dist: {pt[8]:.4f}m")

# Check compute_trajectory_collision_costs
kin = twin.kinematics if hasattr(twin, 'kinematics') else None
if kin is None:
    from drema.controller.franka_kinematics import FrankaKinematics
    kin = FrankaKinematics(base_position=np.array([-0.309, 0.0, 0.82]))

Q = np.tile(q_ik[None, None, :], (10, 15, 1))
QD = np.zeros_like(Q)
cp, gvm = twin.compute_trajectory_collision_costs(Q, QD, 0.02, 0.15, 15.0, 0.8, kin)
print("Trajectory collision costs max cp:", np.max(cp), "max gvm:", np.max(gvm))

twin.shutdown()
