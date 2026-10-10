#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from drema.controller.franka_kinematics import FrankaKinematics

kin = FrankaKinematics()
q = np.array([+0.042, +0.929, +0.011, -1.427, +0.016, +1.231, +0.781], dtype=np.float32)

J_ana = kin.geometric_jacobian(q, link_idx=-1)[:3, :]

eps = 1e-4
J_num = np.zeros((3, 7), dtype=np.float32)
for i in range(7):
    qp = q.copy()
    qm = q.copy()
    qp[i] += eps
    qm[i] -= eps
    pp, _ = kin.forward_kinematics_ee(qp)
    pm, _ = kin.forward_kinematics_ee(qm)
    J_num[:, i] = (pp - pm) / (2.0 * eps)

max_diff = np.max(np.abs(J_ana - J_num))
print("=" * 60)
print("FRANKA KINEMATICS LIVE VERIFICATION:")
print("=" * 60)
print(f"Max difference analytical vs numerical: {max_diff:.8f}")
print("SUCCESS:", max_diff < 1e-3)
print("=" * 60)
