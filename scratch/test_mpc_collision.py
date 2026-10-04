import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.controller.mpc_controller import MPCController
from drema.communication.proto import drema_comm_pb2

twin = PyBulletDigitalTwin(visualize=False, table_z=0.75, tracking_mode="constraint")
twin.load_robot(base_position=(-0.309, 0.0, 0.82))

mesh_path = os.path.abspath("assets/scanned_meshes/tunnel_obstacle_0.obj")
obstacle_pos = (0.378, -0.08, 0.813)
body_id = twin.spawn_scanned_mesh_obstacle(
    mesh_path=mesh_path,
    initial_pos=obstacle_pos,
    name="tunnel_obstacle",
    mass=1.0,
    obj_id=84
)

mpc = MPCController(digital_twin=twin)

# Franka Panda joint positions near target
q_ik = twin.calculate_inverse_kinematics((0.38, 0.02, 0.80))
twin.sync_robot_state(q_ik)
twin.step()

robot_state = drema_comm_pb2.RobotState(
    timestamp=0.0,
    timestep=100,
    joint_positions=list(q_ik),
    joint_velocities=[0.0] * 7,
    task_active=True,
    target_available=True,
    target_pose=[0.38, 0.02, 0.80, 0.0, 0.0, 0.0, 1.0],
    robot_base_pos=[-0.309, 0.0, 0.82]
)

action = mpc.compute_action(robot_state)
print("Action status message:", action.status_message)
print("Action velocities:", np.round(action.joint_velocities, 4))
print("Action safety stop:", action.safety_stop)

twin.shutdown()
