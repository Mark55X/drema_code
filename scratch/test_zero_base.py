import pybullet as p
import pybullet_data
import os

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

urdf_assets = os.path.abspath("assets/franka_panda/panda.urdf")
robot_assets_zero = p.loadURDF(urdf_assets, [0.0, 0.0, 0.0], useFixedBase=True)

print("Zero-base Link 0/base pos:", p.getBasePositionAndOrientation(robot_assets_zero)[0])
print("Zero-base Link 1 (joint 0) world pos:", p.getLinkState(robot_assets_zero, 0)[0])
print("Zero-base Link 7 (wrist) world pos:", p.getLinkState(robot_assets_zero, 6)[0])

p.disconnect()
