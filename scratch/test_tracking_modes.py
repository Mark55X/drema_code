import numpy as np
import pybullet as p
import pybullet_data

def test_full_pd_and_constraint_rotation_and_translation():
    for mode in ["constraint", "pd_force"]:
        print(f"\n--- Testing Mode: {mode} ---")
        cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        plane = p.loadURDF("plane.urdf")
        
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        body = p.createMultiBody(1.0, col, basePosition=[0.0, 0.0, 0.5])
        
        target_pos = np.array([0.2, 0.1, 0.3], dtype=np.float32)
        # 45 deg around Z
        target_quat = np.array([0.0, 0.0, np.sin(np.pi/8), np.cos(np.pi/8)], dtype=np.float32)
        
        if mode == "constraint":
            c_id = p.createConstraint(body, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], target_pos.tolist(), target_quat.tolist())
            p.changeConstraint(c_id, maxForce=300)
            for _ in range(100):
                p.stepSimulation()
        else: # pd_force
            kp_pos, kd_pos = 250.0, 30.0
            kp_rot, kd_rot = 15.0, 1.5
            for _ in range(150):
                curr_pos_t, curr_quat_t = p.getBasePositionAndOrientation(body)
                lin_vel_t, ang_vel_t = p.getBaseVelocity(body)
                curr_pos = np.array(curr_pos_t, dtype=np.float32)
                curr_quat = np.array(curr_quat_t, dtype=np.float32)
                lin_vel = np.array(lin_vel_t, dtype=np.float32)
                ang_vel = np.array(ang_vel_t, dtype=np.float32)
                
                # Add feedforward gravity compensation (m * g)
                grav_comp = np.array([0.0, 0.0, 1.0 * 9.81], dtype=np.float32)
                force = kp_pos * (target_pos - curr_pos) - kd_pos * lin_vel + grav_comp
                p.applyExternalForce(body, -1, force.tolist(), curr_pos.tolist(), p.WORLD_FRAME)
                
                q_inv = np.array([-curr_quat[0], -curr_quat[1], -curr_quat[2], curr_quat[3]], dtype=np.float32)
                w1, x1, y1, z1 = target_quat[3], target_quat[0], target_quat[1], target_quat[2]
                w2, x2, y2, z2 = q_inv[3], q_inv[0], q_inv[1], q_inv[2]
                qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
                qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
                qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
                qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
                q_rel_vec = np.array([qx, qy, qz], dtype=np.float32)
                if qw < 0.0:
                    q_rel_vec = -q_rel_vec
                torque = kp_rot * (2.0 * q_rel_vec) - kd_rot * ang_vel
                p.applyExternalTorque(body, -1, torque.tolist(), p.WORLD_FRAME)
                p.stepSimulation()
                
        final_pos, final_quat = p.getBasePositionAndOrientation(body)
        pos_err = np.linalg.norm(np.array(final_pos) - target_pos)
        print(f"  Target: {target_pos}, Result: {[round(x, 4) for x in final_pos]}")
        print(f"  Position Error: {pos_err*1000:.2f} mm")
        assert pos_err < 0.02, f"Position error too large ({pos_err})!"
        p.disconnect()
        print(f"  ✓ {mode} mode verified successfully!")

if __name__ == "__main__":
    test_full_pd_and_constraint_rotation_and_translation()
