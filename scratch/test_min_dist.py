import pybullet as p
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

twin = PyBulletDigitalTwin(visualize=False, table_z=0.75, tracking_mode="constraint")
twin.load_robot(base_position=(-0.309, 0.0, 0.82))

mesh_path = os.path.abspath("assets/franka_panda/objs/robot_base.obj")
obstacle_pos = (0.378, 0.0, 0.813)
body_id = twin.spawn_scanned_mesh_obstacle(
    mesh_path=mesh_path,
    initial_pos=obstacle_pos,
    name="tunnel_obstacle",
    mass=1.0
)

# Start robot in default homing (away from obstacle)
q_init = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
twin.sync_robot_state(q_init)
# Step physics so collision tree is built at q_init
twin.step()
print("At q_init, min_d:", twin.get_min_obstacle_distance())

# Now robot moves to reach target near obstacle
q_ik = twin.calculate_inverse_kinematics((0.38, 0.02, 0.80))
twin.sync_robot_state(q_ik)
# NOTICE: We only call twin.step() just like in on_request_action!
twin.step()

# Now check min_dist WITHOUT performCollisionDetection
pts_direct = p.getClosestPoints(bodyA=twin.robot_id, bodyB=body_id, distance=0.5)
print("After twin.step(), closest points count:", len(pts_direct))
print("twin.get_min_obstacle_distance():", twin.get_min_obstacle_distance())

twin.shutdown()
