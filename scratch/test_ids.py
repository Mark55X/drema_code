import pybullet as p
import pybullet_data
import os
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf") # ID 0

# Spawn table (ID 1)
col_table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.375])
table_id = p.createMultiBody(0.0, col_table, -1, [0.0, 0.0, 0.375]) # ID 1

# Spawn tunnel obstacle (ID 2)
mesh_path = os.path.abspath("assets/scanned_meshes/tunnel_obstacle_0.obj")
col_mesh = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path)
body_mesh = p.createMultiBody(1.0, col_mesh, -1, [0.38, -0.08, 0.813]) # ID 2

# Spawn Franka Panda (ID 3)
robot_id = p.loadURDF("franka_panda/panda.urdf", [-0.309, 0.0, 0.82], useFixedBase=True) # ID 3

print(f"Table ID: {table_id}, Obstacle ID: {body_mesh}, Robot ID: {robot_id}")

p.performCollisionDetection()
pts = p.getClosestPoints(robot_id, body_mesh, 2.0)
print(f"Closest points count (dist 2.0): {len(pts)}")
if pts:
    print(f"Min dist: {min(pt[8] for pt in pts)}")

p.disconnect()
