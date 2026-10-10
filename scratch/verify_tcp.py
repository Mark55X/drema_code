import sys
import numpy as np

# Add drema_code to sys.path
sys.path.insert(0, '/home/marcotefani/master-thesis/drema_code')

from drema.controller.franka_kinematics import FrankaKinematics

# Test with current EE_OFFSET and fingertip EE_OFFSET
q_log = np.array([+0.292, +1.166, -0.200, -1.129, +0.035, +1.130, +0.792], dtype=np.float32)
tgt = np.array([0.38, 0.02, 0.80], dtype=np.float32)

print("=== VERIFYING TCP OFFSET ===")
kin_orig = FrankaKinematics(base_position=np.array([-0.309, 0.0, 0.820], dtype=np.float32))
p_ee_orig, _ = kin_orig.forward_kinematics_ee(q_log)
print("Original FrankaKinematics EE (z_offset = 0.1034m):")
print("  EE Pos:", p_ee_orig)
print("  Distance to Target:", np.linalg.norm(p_ee_orig - tgt))

kin_tip = FrankaKinematics(base_position=np.array([-0.309, 0.0, 0.820], dtype=np.float32))
kin_tip.EE_OFFSET = np.array([0.0, 0.0, 0.2170], dtype=np.float32)
p_ee_tip, _ = kin_tip.forward_kinematics_ee(q_log)
print("\nAdjusted Fingertip EE (z_offset = 0.2170m):")
print("  Fingertip Pos:", p_ee_tip)
print("  Distance to Target:", np.linalg.norm(p_ee_tip - tgt))
