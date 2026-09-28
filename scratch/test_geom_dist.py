import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/franka_panda/panda.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
fromto = np.zeros(6, dtype=np.float64)
dist = mujoco.mj_geomDistance(m, d, 0, 1, 1.0, fromto)
print('Distance:', dist, 'fromto:', fromto)
