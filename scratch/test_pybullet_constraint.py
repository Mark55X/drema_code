import pybullet as p
import time

client = p.connect(p.DIRECT)
p.setGravity(0, 0, 0) # No gravity for clean tracking test

# Create obstacle
col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
body_id = p.createMultiBody(1.0, col_id, -1, [0.0, 0.0, 0.0], [0, 0, 0, 1])

cid = p.createConstraint(
    parentBodyUniqueId=body_id,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0.0, 0.0, 0.0],
    parentFramePosition=[0.0, 0.0, 0.0],
    childFramePosition=[0.0, 0.0, 0.0],
    childFrameOrientation=[0, 0, 0, 1]
)
p.changeConstraint(cid, maxForce=300.0)

pos_init, _ = p.getBasePositionAndOrientation(body_id)
print(f"Initial pos: {pos_init}")

# Try moving constraint to [1.0, 2.0, 3.0]
new_pos = [1.0, 2.0, 3.0]
p.changeConstraint(cid, jointChildPivot=new_pos, jointChildFrameOrientation=[0, 0, 0, 1], maxForce=300.0)

# Step simulation 10 times
for _ in range(10):
    p.stepSimulation()

pos_after_steps, _ = p.getBasePositionAndOrientation(body_id)
print(f"Pos after 10 steps: {pos_after_steps}")

# Step simulation 240 times (1 second of physics at 240Hz)
for _ in range(240):
    p.stepSimulation()

pos_after_1sec, _ = p.getBasePositionAndOrientation(body_id)
print(f"Pos after 240 steps: {pos_after_1sec}")

p.disconnect()
