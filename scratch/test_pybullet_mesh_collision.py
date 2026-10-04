import os
import pybullet as p
import pybullet_data

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# Test object at [0.1, 0.0, 0.5]
col_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
body_sphere = p.createMultiBody(1.0, col_sphere, -1, [0.1, 0.0, 0.5])

# Case 1: Without performCollisionDetection
pts_no_detect = p.getClosestPoints(robot_id, body_sphere, 0.5)
print(f"Without performCollisionDetection count: {len(pts_no_detect)}")

# Case 2: With performCollisionDetection
p.performCollisionDetection()
pts_with_detect = p.getClosestPoints(robot_id, body_sphere, 0.5)
print(f"With performCollisionDetection count: {len(pts_with_detect)}")

p.disconnect()
