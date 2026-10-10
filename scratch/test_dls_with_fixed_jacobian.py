#!/usr/bin/env python3
import numpy as np

MDH_PARAMS = [
    (0.0,      0.0,         0.333, 0.0),        # Joint 1
    (0.0,     -np.pi / 2.0, 0.0,   0.0),        # Joint 2
    (0.0,      np.pi / 2.0, 0.316, 0.0),        # Joint 3
    (0.0825,   np.pi / 2.0, 0.0,   0.0),        # Joint 4
    (-0.0825, -np.pi / 2.0, 0.384, 0.0),        # Joint 5
    (0.0,      np.pi / 2.0, 0.0,   0.0),        # Joint 6
    (0.088,    np.pi / 2.0, 0.107, 0.0),        # Joint 7
]
EE_OFFSET = np.array([0.00139, 0.00144, 0.11003], dtype=np.float32)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)

def dh(a, alpha, d, theta):
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_al, s_al = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c_th, -s_th, 0.0, a],
        [s_th*c_al, c_th*c_al, -s_al, -s_al*d],
        [s_th*s_al, c_th*s_al, c_al, c_al*d],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

def fk_all(q):
    T = np.eye(4, dtype=np.float32)
    transforms = [T.copy()]
    for i in range(7):
        a, al, d, th_off = MDH_PARAMS[i]
        T = T @ dh(a, al, d, q[i] + th_off)
        transforms.append(T.copy())
    T_ee = T.copy()
    T_ee[:3, 3] += T[:3, :3] @ EE_OFFSET
    transforms.append(T_ee)
    return transforms

def jacobian_fixed(q):
    all_T = fk_all(q)
    p_ee = all_T[-1][:3, 3]
    J = np.zeros((6, 7), dtype=np.float32)
    for i in range(7):
        a, al, d, th_off = MDH_PARAMS[i]
        cal, sal = np.cos(al), np.sin(al)
        Rx_Tx = np.array([
            [1.0, 0.0,  0.0,  a],
            [0.0, cal, -sal, 0.0],
            [0.0, sal,  cal, 0.0],
            [0.0, 0.0,  0.0, 1.0]
        ], dtype=np.float32)
        T_prime = all_T[i] @ Rx_Tx
        z_i = T_prime[:3, 2]
        p_i = T_prime[:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
    return J

def jacobian_buggy(q):
    all_T = fk_all(q)
    p_ee = all_T[-1][:3, 3]
    J = np.zeros((6, 7), dtype=np.float32)
    for i in range(7):
        T_prev = all_T[i]
        z_i = T_prev[:3, 2]
        p_i = T_prev[:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
    return J

def solve_ik(q_init, target_pos_rel, use_fixed_jacobian=True, max_iters=40):
    q = q_init.copy()
    for it in range(max_iters):
        all_T = fk_all(q)
        current_p = all_T[-1][:3, 3]
        err = target_pos_rel - current_p
        dist = np.linalg.norm(err)
        if dist < 2e-3:
            return q, True, it, dist
        J = jacobian_fixed(q)[:3, :] if use_fixed_jacobian else jacobian_buggy(q)[:3, :]
        JJT = J @ J.T
        dq = J.T @ np.linalg.inv(JJT + 0.01 * np.eye(3, dtype=np.float32)) @ err
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    all_T = fk_all(q)
    dist = np.linalg.norm(target_pos_rel - all_T[-1][:3, 3])
    return q, dist < 2e-3, max_iters, dist

# Target relative to base:
# base_pos = [-0.26796, -0.00639, 0.75003]
# target_world = [0.38, 0.02, 0.80]
target_rel = np.array([0.38 - (-0.26796), 0.02 - (-0.00639), 0.80 - 0.75003], dtype=np.float32)
q_init = np.array([+0.000, +0.176, -0.000, -0.873, +0.000, +1.221, +0.785], dtype=np.float32)

print("=" * 60)
print("TESTING IK CONVERGENCE TO TARGET")
print("=" * 60)
q_sol_buggy, succ_buggy, it_buggy, dist_buggy = solve_ik(q_init, target_rel, use_fixed_jacobian=False)
print(f"BUGGY Jacobian IK:  Success={succ_buggy}, Iterations={it_buggy}, Final Error={dist_buggy*1000:.2f} mm")

q_sol_fixed, succ_fixed, it_fixed, dist_fixed = solve_ik(q_init, target_rel, use_fixed_jacobian=True)
print(f"FIXED Jacobian IK:  Success={succ_fixed}, Iterations={it_fixed}, Final Error={dist_fixed*1000:.2f} mm")
print("Fixed IK Solution Joint Angles:\n", np.round(q_sol_fixed, 4))
print("=" * 60)
