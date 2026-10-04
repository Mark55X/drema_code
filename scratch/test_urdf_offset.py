import pybullet as p
import pybullet_data
import os

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# What is loaded in pybullet_digital_twin:
# 1. Using "franka_panda/panda.urdf" (from pybullet_data) vs "assets/franka_panda/panda.urdf"
urdf_assets = os.path.abspath("assets/franka_panda/panda.urdf")
print("assets urdf exists:", os.path.exists(urdf_assets))

# Test loading assets/franka_panda/panda.urdf at base_pos=[-0.309, 0.0, 0.82]
robot_assets = p.loadURDF(urdf_assets, [-0.309, 0.0, 0.82], useFixedBase=True)
base_link_state = p.getBasePositionAndOrientation(robot_assets)
print("Assets URDF base pos:", base_link_state[0])

# Check Link 0 / Link 1 world positions
link1_state = p.getLinkState(robot_assets, 0)
print("Assets URDF Link 1 (joint 0) world pos:", link1_state[0])

# Check Link 7 (end effector / wrist) world pos in default joint angles
link7_state = p.getLinkState(robot_assets, 6)
print("Assets URDF Link 7 (wrist) world pos:", link7_state[0])

# Now compare with pybullet_data's standard "franka_panda/panda.urdf"
robot_pybullet = p.loadURDF("franka_panda/panda.urdf", [-0.309, 0.0, 0.82], useFixedBase=True)
pb_link1_state = p.getLinkState(robot_pybullet, 0)
print("PyBullet standard panda Link 1 world pos:", pb_link1_state[0])
pb_link7_state = p.getLinkState(robot_pybullet, 6)
print("PyBullet standard panda Link 7 world pos:", pb_link7_state[0])

p.disconnect()
