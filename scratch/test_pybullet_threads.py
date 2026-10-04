import pybullet as p
import pybullet_data
import threading
import time
import numpy as np

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
body_id = p.createMultiBody(1.0, col_id, -1, [0.3, 0.0, 0.5])

errors = []
stop_flag = False

def perception_thread():
    step = 0
    while not stop_flag:
        try:
            step += 1
            y = 0.2 * np.sin(step * 0.1)
            p.resetBasePositionAndOrientation(body_id, [0.3, y, 0.5], [0, 0, 0, 1])
            time.sleep(0.01)
        except Exception as e:
            errors.append(f"Perception thread error: {e}")

def mpc_thread():
    for _ in range(50):
        try:
            for j in range(7):
                p.resetJointState(robot_id, j, np.random.randn() * 0.1)
            p.performCollisionDetection()
            pts = p.getClosestPoints(robot_id, body_id, 0.5)
            time.sleep(0.01)
        except Exception as e:
            errors.append(f"MPC thread error: {e}")

t1 = threading.Thread(target=perception_thread)
t2 = threading.Thread(target=mpc_thread)

t1.start()
t2.start()

t2.join()
stop_flag = True
t1.join()

print("PyBullet multi-thread test errors count:", len(errors))
for err in errors[:5]:
    print("  ", err)

p.disconnect()
