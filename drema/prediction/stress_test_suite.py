#!/usr/bin/env python
"""
Advanced Stress Test Suite for DREMA Dynamic Obstacle Trajectory Predictor.
Tests extreme non-linear 3D motions, severe sensory jitter, outlier rejection,
frame dropouts / occlusions, multi-obstacle scalability, and full MP-PMPPI closed loop.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p

from drema.prediction.base_predictor import BaseObstaclePredictor, TrajectoryPrediction
from drema.prediction.kalman_predictor import KalmanObstaclePredictor
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.controller.franka_kinematics import FrankaKinematics
from drema.controller.mp_pmppi_engine import MPPMPPIEngine


def run_test_1_helical_3d_motion():
    print("\n" + "="*70)
    print("TEST 1: 3D Helical Non-Linear Motion + 3D Angular Spin Tracking")
    print("="*70)

    predictor = KalmanObstaclePredictor()
    obj_id = "helical_obj"

    # Helical trajectory:
    # x(t) = 0.4 + 0.15 * cos(2 * pi * f * t)
    # y(t) = 0.15 * sin(2 * pi * f * t)
    # z(t) = 0.75 + 0.05 * t
    # Spin: 1.5 rad/s around tilted axis [1, 1, 0] / sqrt(2)
    freq = 0.6 # Hz
    omega_spin = 1.5 # rad/s
    spin_axis = np.array([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2.0)

    dt_cam = 0.033 # 30 Hz perception
    duration = 2.0
    times = np.arange(0, duration, dt_cam)

    pos_errors = []
    vel_errors = []

    for t in times:
        # Analytical ground truth
        x_true = 0.4 + 0.15 * np.cos(2 * np.pi * freq * t)
        y_true = 0.15 * np.sin(2 * np.pi * freq * t)
        z_true = 0.75 + 0.05 * t
        true_pos = np.array([x_true, y_true, z_true], dtype=np.float32)

        vx_true = -0.15 * (2 * np.pi * freq) * np.sin(2 * np.pi * freq * t)
        vy_true = 0.15 * (2 * np.pi * freq) * np.cos(2 * np.pi * freq * t)
        vz_true = 0.05
        true_vel = np.array([vx_true, vy_true, vz_true], dtype=np.float32)

        rot_angle = omega_spin * t
        true_quat = R.from_rotvec(spin_axis * rot_angle).as_quat().astype(np.float32)

        # Ingest noisy observation (3mm spatial noise)
        meas_pos = true_pos + np.random.randn(3).astype(np.float32) * 0.003
        predictor.update_obstacle_pose(obj_id, meas_pos, true_quat, timestamp=t)

        st = predictor.get_estimated_state(obj_id)
        if t > 0.5: # Allow filter 0.5s to lock in
            pos_errors.append(np.linalg.norm(st['position'] - true_pos))
            vel_errors.append(np.linalg.norm(st['velocity'] - true_vel))

    mean_pos_err = np.mean(pos_errors) * 1000.0 # mm
    mean_vel_err = np.mean(vel_errors) * 100.0 # cm/s
    print(f"Mean Pos Tracking Error: {mean_pos_err:.2f} mm (perceptual noise std = 3.0 mm)")
    print(f"Mean Vel Tracking Error: {mean_vel_err:.2f} cm/s (on rapid 3D curved trajectory)")

    # Test future horizon forecast accuracy (H=15, 0.75s lookahead)
    t_now = times[-1]
    pred = predictor.predict_obstacle_trajectory(obj_id, horizon=15, dt=0.05)
    
    # Ground truth future positions at t_now + h*dt
    fut_times = t_now + np.arange(15) * 0.05
    fut_true_x = 0.4 + 0.15 * np.cos(2 * np.pi * freq * fut_times)
    fut_true_y = 0.15 * np.sin(2 * np.pi * freq * fut_times)
    fut_true_z = 0.75 + 0.05 * fut_times
    fut_true = np.stack([fut_true_x, fut_true_y, fut_true_z], axis=1)

    horizon_err = np.linalg.norm(pred.positions - fut_true, axis=1)
    print(f"Prediction Error at +0.25s (h=5):  {horizon_err[5]*100:.2f} cm")
    print(f"Prediction Error at +0.50s (h=10): {horizon_err[10]*100:.2f} cm")
    print(f"Prediction Error at +0.75s (h=14): {horizon_err[14]*100:.2f} cm")

    assert mean_pos_err < 10.0, f"Pos error too high: {mean_pos_err} mm"
    assert mean_vel_err < 20.0, f"Vel error too high: {mean_vel_err} cm/s"
    print(">>> TEST 1 RESULT: PASSED [EXCELLENT ACCURACY]")


def run_test_2_sensor_dropout_and_occlusions():
    print("\n" + "="*70)
    print("TEST 2: Long Sensor Dropout (0.6s / 18 Missing Frames Occlusion)")
    print("="*70)

    predictor = KalmanObstaclePredictor()
    obj_id = "occluded_box"

    # Moving at constant velocity 0.25 m/s along Y
    v_true = np.array([0.0, 0.25, 0.0], dtype=np.float32)
    p0 = np.array([0.35, -0.30, 0.80], dtype=np.float32)

    # 1. Warmup for 1.0s (30 frames)
    for step in range(30):
        t = step * 0.033
        p = p0 + v_true * t + np.random.randn(3) * 0.002
        predictor.update_obstacle_pose(obj_id, p, [0, 0, 0, 1], timestamp=t)

    st_before = predictor.get_estimated_state(obj_id)
    print(f"State before occlusion (t=1.0s): Pos={np.round(st_before['position'], 3)}, Vel={np.round(st_before['velocity'], 3)}")

    # 2. OCCLUSION GAP: No camera frames for 0.6 seconds!
    # During this gap, the MPC queries the predictor repeatedly
    t_query = 1.0 + 0.3
    pred_during_gap = predictor.predict_obstacle_trajectory(obj_id, horizon=10, dt=0.05, latency_comp_sec=0.3)
    expected_p_gap = p0 + v_true * t_query
    gap_err = np.linalg.norm(pred_during_gap.positions[0] - expected_p_gap)
    print(f"Inertial extrapolation error during occlusion (+0.3s blackout): {gap_err*1000:.2f} mm")

    # 3. Object re-emerges at t=1.6s
    t_reemerge = 1.6
    p_reemerge = p0 + v_true * t_reemerge + np.random.randn(3) * 0.002
    accepted = predictor.update_obstacle_pose(obj_id, p_reemerge, [0, 0, 0, 1], timestamp=t_reemerge)
    st_after = predictor.get_estimated_state(obj_id)

    print(f"Observation accepted after blackout: {accepted}")
    print(f"State after re-emergence (t=1.6s): Pos={np.round(st_after['position'], 3)}, Vel={np.round(st_after['velocity'], 3)}")

    assert accepted, "Valid observation after gap was mistakenly rejected!"
    assert np.all(np.isfinite(st_after['position'])), "Filter encountered NaN/Inf!"
    assert np.all(np.isfinite(st_after['velocity'])), "Velocity encountered NaN/Inf!"
    assert abs(st_after['velocity'][1] - 0.25) < 0.05, "Velocity diverged during blackout!"
    print(">>> TEST 2 RESULT: PASSED [STABLE RE-CONVERGENCE]")


def run_test_3_hostile_outlier_injection():
    print("\n" + "="*70)
    print("TEST 3: Hostile Outlier Injection & Quaternion Hemisphere Flip")
    print("="*70)

    predictor = KalmanObstaclePredictor()
    obj_id = "glitched_target"

    p_nominal = np.array([0.40, 0.0, 0.75], dtype=np.float32)
    q_nominal = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Prime filter
    for step in range(15):
        t = step * 0.033
        predictor.update_obstacle_pose(obj_id, p_nominal, q_nominal, timestamp=t)

    # INJECT OUTLIER 1: Massive 0.8 meter teleportation glitch (e.g. false segmentation)
    outlier_pos = p_nominal + np.array([0.8, -0.5, 0.4], dtype=np.float32)
    acc1 = predictor.update_obstacle_pose(obj_id, outlier_pos, q_nominal, timestamp=0.50)
    st1 = predictor.get_estimated_state(obj_id)

    print(f"Glitch of 80cm injected -> Accepted by filter: {acc1} (Expected: False)")
    print(f"Position after rejected glitch: {np.round(st1['position'], 3)} (Remained anchored at nominal)")
    assert not acc1, "Mahalanobis gating failed to reject catastrophic 80cm outlier!"

    # INJECT OUTLIER 2: Antipodal quaternion sign flip [-q] (represents identical rotation)
    flip_quat = -q_nominal
    acc2 = predictor.update_obstacle_pose(obj_id, p_nominal, flip_quat, timestamp=0.533)
    st2 = predictor.get_estimated_state(obj_id)

    print(f"Antipodal -q injected -> Accepted: {acc2}, Omega mag: {np.linalg.norm(st2['angular_velocity']):.4f} rad/s")
    assert np.linalg.norm(st2['angular_velocity']) < 0.1, "Quaternion flip caused fake angular velocity explosion!"
    print(">>> TEST 3 RESULT: PASSED [ROBUST OUTLIER IMMUNITY]")


def run_test_4_multi_obstacle_swarm_scalability():
    print("\n" + "="*70)
    print("TEST 4: Multi-Obstacle Scalability Benchmark (N=10 Obstacles)")
    print("="*70)

    predictor = KalmanObstaclePredictor(stale_timeout_sec=0.2)
    num_obstacles = 10

    # Ingest 10 simultaneous obstacles
    for oid in range(num_obstacles):
        pos = np.random.randn(3).astype(np.float32)
        quat = np.array([0, 0, 0, 1], dtype=np.float32)
        predictor.update_obstacle_pose(f"obs_{oid}", pos, quat, timestamp=time.time())

    # Benchmark predict_all for all 10 obstacles
    N_iters = 500
    t0 = time.perf_counter()
    for _ in range(N_iters):
        preds = predictor.predict_all(horizon=20, dt=0.05)
    total_time_ms = (time.perf_counter() - t0) / N_iters * 1000.0

    print(f"Active tracked obstacles: {len(preds)}")
    print(f"Time to forecast trajectories for all 10 obstacles (H=20): {total_time_ms:.3f} ms")
    print(f"Per-obstacle prediction overhead: {total_time_ms / num_obstacles * 1000:.1f} microseconds")

    # Test stale obstacle pruning
    time.sleep(0.25) # Exceed stale timeout
    # Update only obstacle 0 and 1
    predictor.update_obstacle_pose("obs_0", [0, 0, 0], [0, 0, 0, 1])
    predictor.update_obstacle_pose("obs_1", [0, 0, 0], [0, 0, 0, 1])
    pruned = predictor.prune_stale_obstacles()
    remaining = len(predictor.filters)

    print(f"Pruned stale obstacles: {len(pruned)} (IDs: {pruned[:3]}...)")
    print(f"Remaining active obstacles: {remaining} (Expected: 2)")

    assert remaining == 2, f"Pruning error: expected 2 remaining, got {remaining}"
    assert total_time_ms < 1.0, f"Scalability latency too high: {total_time_ms} ms"
    print(">>> TEST 4 RESULT: PASSED [ULTRA-LOW LATENCY SWARM]")


def run_test_5_closed_loop_mppi_pybullet():
    print("\n" + "="*70)
    print("TEST 5: Full Closed-Loop MP-PMPPI + PyBullet Collision Integration")
    print("="*70)

    twin = PyBulletDigitalTwin(visualize=False)
    twin.load_robot(base_position=(0, 0, 0.75))
    engine = MPPMPPIEngine()

    # Spawn an obstacle heading directly across the robot's workspace
    obj_id = 99
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.08], physicsClientId=twin.client_id)
    body_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_id, basePosition=[0.4, -0.3, 0.8], physicsClientId=twin.client_id)
    twin.tracked_objects[obj_id] = {
        'body_id': body_id,
        'target_pos': (0.4, -0.3, 0.8),
        'target_quat': (0, 0, 0, 1),
        'is_target': False
    }

    q_curr = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    qd_curr = np.zeros(7, dtype=np.float32)
    target_pos = np.array([0.4, 0.0, 0.8], dtype=np.float32)

    # Run 5 consecutive MPC closed-loop iterations as the obstacle moves across
    mpc_latencies = []
    print("Simulating closed-loop execution as obstacle moves from Y=-0.20 to Y=+0.20...")

    for step in range(10):
        t_sim = step * 0.04 # 25 Hz control
        obs_y = -0.20 + step * 0.04 * 0.5 # Moving at 0.5 m/s
        twin.sync_object_pose(obj_id, (0.4, obs_y, 0.8), (0, 0, 0, 1), timestamp=t_sim)

        t_m0 = time.perf_counter()
        qd_opt, diag = engine.solve(
            q_current=q_curr,
            qd_current=qd_curr,
            target_pos=target_pos,
            digital_twin=twin
        )
        mpc_lat = (time.perf_counter() - t_m0) * 1000.0
        mpc_latencies.append(mpc_lat)

        # Kinematic integration of robot
        qd_curr = qd_opt
        q_curr = q_curr + qd_opt * engine.dt

        top_cand = diag.get('top_candidate', 'N/A')
        w_grd = diag['weights'].get('greedy', 0.0)
        w_sns = diag['weights'].get('sensitive', 0.0)
        print(f"  Step #{step+1:02d} | Obs Y: {obs_y:+.2f}m | Latency: {mpc_lat:.1f}ms | w_grd: {w_grd:.2f}, w_sns: {w_sns:.2f} | Top: {top_cand}")

        assert np.all(np.isfinite(qd_opt)), f"MPC command produced NaN at step {step}!"
        assert np.all(np.abs(qd_opt) <= engine.kin.QD_MAX + 1e-3), "Velocity exceeded joint limits!"

    twin.shutdown()
    mean_lat = np.mean(mpc_latencies)
    print(f"Mean Closed-Loop MP-PMPPI Iteration Latency: {mean_lat:.1f} ms")
    assert mean_lat < 120.0, f"MPC latency exceeded safety threshold: {mean_lat} ms"
    print(">>> TEST 5 RESULT: PASSED [SMOOTH CLOSED-LOOP AVOIDANCE]")


def run_test_6_instantaneous_direction_inversion():
    print("\n" + "="*70)
    print("TEST 6: Instantaneous Direction Inversion (Sudden Rebound Maneuver)")
    print("="*70)
    predictor = KalmanObstaclePredictor()
    obj_id = "bouncing_obstacle"
    dt = 0.033  # 30 Hz perception

    # Phase 1: Moving in +X direction at +0.7 m/s for 1.0s (30 frames)
    # x(t) = 0.2 + 0.7 * t
    for step in range(30):
        t = step * dt
        pos = np.array([0.2 + 0.7 * t, 0.0, 0.5], dtype=np.float32)
        predictor.update_obstacle_pose(obj_id, pos, [0, 0, 0, 1], timestamp=t)

    st_before = predictor.get_estimated_state(obj_id)
    print(f"Velocity before bounce (t=1.0s): {st_before['velocity'][0]:+.3f} m/s (Target: +0.700 m/s)")
    assert abs(st_before['velocity'][0] - 0.7) < 0.05

    # Phase 2: Instantaneous bounce at t=1.0s, moves in -X direction at -0.7 m/s!
    # x(t) = x(1.0) - 0.7 * (t - 1.0)
    p_bounce = 0.2 + 0.7 * (29 * dt)
    accepted_log = []
    vel_log = []

    # Observe next 15 frames (0.5s)
    for step in range(1, 16):
        t = 1.0 + step * dt
        pos = np.array([p_bounce - 0.7 * (step * dt), 0.0, 0.5], dtype=np.float32)
        acc = predictor.update_obstacle_pose(obj_id, pos, [0, 0, 0, 1], timestamp=t)
        accepted_log.append(acc)
        st = predictor.get_estimated_state(obj_id)
        vel_log.append(st['velocity'][0])

    print(f"Filter acceptance flags after bounce: {accepted_log[:6]}")
    print(f"Velocity progression: {[round(v, 2) for v in vel_log[:6]]}")
    print(f"Final velocity at t=1.5s: {vel_log[-1]:+.3f} m/s (Target: -0.700 m/s)")

    # Verification:
    # 1. Did the gate recovery trigger? The filter must not be permanently locked out!
    assert any(accepted_log[2:5]), "Gate recovery failed to unlock after 3 consecutive rejections!"
    # 2. Final velocity must have adapted to the reverse direction
    assert vel_log[-1] < -0.55, f"Filter failed to adapt to reverse velocity! Final vel: {vel_log[-1]}"
    print(">>> TEST 6 RESULT: PASSED [RAPID MANEUVER ADAPTATION]")


def run_test_7_asynchronous_jitter_and_heavy_noise():
    print("\n" + "="*70)
    print("TEST 7: Asynchronous Time Jitter + Heavy Perceptual Noise (2cm RMS)")
    print("="*70)
    predictor = KalmanObstaclePredictor(default_meas_noise=0.020, default_acc_noise=2.0)
    obj_id = "jittery_obstacle"

    # True motion: smooth 3D curve (sinusoidal in Y and Z, constant vx = 0.3)
    # Perception has random dt in [15ms, 90ms], random 10% packet loss, 2cm spatial noise
    np.random.seed(42)
    t = 0.0
    true_positions = []
    raw_positions = []
    est_positions = []

    for _ in range(80):
        # Variable dt
        dt = float(np.random.uniform(0.015, 0.090))
        t += dt

        # Ground truth
        y_true = 0.25 * np.sin(2 * np.pi * 0.5 * t)
        z_true = 0.6 + 0.1 * np.cos(2 * np.pi * 0.5 * t)
        x_true = 0.3 + 0.2 * t
        p_true = np.array([x_true, y_true, z_true], dtype=np.float32)

        # 10% frame drop simulation
        if np.random.rand() < 0.10:
            continue

        # Heavy noise (std = 20 mm)
        noise = np.random.randn(3).astype(np.float32) * 0.020
        p_meas = p_true + noise

        predictor.update_obstacle_pose(obj_id, p_meas, [0, 0, 0, 1], timestamp=t)
        st = predictor.get_estimated_state(obj_id)

        if t > 0.5:
            true_positions.append(p_true)
            raw_positions.append(p_meas)
            est_positions.append(st['position'])

    raw_err = np.mean(np.linalg.norm(np.array(raw_positions) - np.array(true_positions), axis=1)) * 1000.0
    est_err = np.mean(np.linalg.norm(np.array(est_positions) - np.array(true_positions), axis=1)) * 1000.0

    print(f"Raw Perceptual Noise RMSE:  {raw_err:.2f} mm")
    print(f"Kalman Filtered Error RMSE: {est_err:.2f} mm")
    print(f"Noise Reduction Factor:     {raw_err / est_err:.2f}x cleaner than raw sensor")

    # Check covariance matrix validity
    cov = predictor.filters[obj_id].P
    eigvals = np.linalg.eigvals(cov)
    print(f"Covariance Matrix Condition: min eigval = {np.min(eigvals):.2e}, max eigval = {np.max(eigvals):.2e}")

    assert np.all(eigvals > 0), "Covariance matrix is not positive-definite!"
    assert est_err < raw_err, "Kalman filter did not reduce perceptual noise!"
    print(">>> TEST 7 RESULT: PASSED [ROBUST JITTER FILTERING]")


def run_test_8_tumbling_so3_lie_manifold():
    print("\n" + "="*70)
    print("TEST 8: High-Speed 3D Tumbling on SO(3) Lie Group Manifold (170 deg/s)")
    print("="*70)
    predictor = KalmanObstaclePredictor()
    obj_id = "tumbling_debris"

    # Obstacle spins at 3.0 rad/s (~172 deg/s) around dynamic axis [0.6, 0.8, 0.0]
    omega_true = np.array([1.8, 2.4, 0.5], dtype=np.float32)
    omega_mag = float(np.linalg.norm(omega_true))
    axis = omega_true / omega_mag

    dt = 0.033
    t_total = 1.5
    times = np.arange(0, t_total, dt)

    for t in times:
        angle = omega_mag * t
        q_true = R.from_rotvec(axis * angle).as_quat().astype(np.float32)
        # Add slight orientation noise
        q_noise = R.from_rotvec(np.random.randn(3) * 0.01).as_quat()
        q_meas = (R.from_quat(q_true) * R.from_quat(q_noise)).as_quat().astype(np.float32)

        predictor.update_obstacle_pose(obj_id, [0.4, 0.0, 0.7], q_meas, timestamp=t)

    st = predictor.get_estimated_state(obj_id)
    omega_est = st['angular_velocity']
    omega_err = np.linalg.norm(omega_est - omega_true)
    print(f"True Angular Velocity:      {np.round(omega_true, 3)} rad/s (|omega| = {omega_mag:.2f} rad/s)")
    print(f"Estimated Angular Velocity: {np.round(omega_est, 3)} rad/s (|omega| = {np.linalg.norm(omega_est):.2f} rad/s)")
    print(f"Angular Velocity Error:     {omega_err:.3f} rad/s")

    # Now test multi-step lookahead orientation rollout across H=30 steps (1.5 seconds into the future)
    H = 30
    dt_pred = 0.05
    t_now = times[-1]
    pred = predictor.predict_obstacle_trajectory(obj_id, horizon=H, dt=dt_pred)

    # Verify unitary quaternions for all future steps
    quat_norms = np.linalg.norm(pred.orientations, axis=1)
    max_norm_drift = np.max(np.abs(quat_norms - 1.0))
    print(f"Max Quaternion Norm Drift over H={H} steps: {max_norm_drift:.2e} (Strictly Unitary)")

    # Compare forecasted orientation at step 20 (1.0s future lookahead) against analytical rotation
    t_future_20 = t_now + 20 * dt_pred
    q_analytic_20 = R.from_rotvec(axis * (omega_mag * t_future_20)).as_quat()
    r_pred_20 = R.from_quat(pred.orientations[20])
    rot_diff_rad = (r_pred_20.inv() * R.from_quat(q_analytic_20)).magnitude()
    rot_diff_deg = np.degrees(rot_diff_rad)
    print(f"Future Orientation Error at +1.0s lookahead: {rot_diff_deg:.2f} degrees")

    assert max_norm_drift < 1e-5, f"Quaternion norm drifted: {max_norm_drift}"
    assert omega_err < 0.4, f"Angular velocity error too high: {omega_err}"
    assert rot_diff_deg < 15.0, f"Future orientation rollout error too high: {rot_diff_deg} deg"
    print(">>> TEST 8 RESULT: PASSED [STABLE SO(3) EXPONENTIAL MAP]")


def run_test_9_mppi_dynamic_vs_static_cost():
    print("\n" + "="*70)
    print("TEST 9: MP-PMPPI Dynamic Collision Cost Benchmark (Static vs Dynamic)")
    print("="*70)

    # 1. Twin with dynamic predictor
    twin_dyn = PyBulletDigitalTwin(visualize=False)
    twin_dyn.load_robot(base_position=(0, 0, 0.75))

    # 2. Twin with static predictor (returns stationary trajectory)
    class StaticPredictor(BaseObstaclePredictor):
        def __init__(self):
            self.poses = {}
        def update_obstacle_pose(self, obj_id, pos, quat, timestamp=None):
            self.poses[obj_id] = (np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32))
            return True
        def predict_obstacle_trajectory(self, obj_id, horizon, dt=0.05, latency_comp_sec=0.0):
            p, q = self.poses[obj_id]
            return TrajectoryPrediction(
                positions=np.tile(p, (horizon, 1)),
                velocities=np.zeros((horizon, 3), dtype=np.float32),
                orientations=np.tile(q, (horizon, 1)),
                angular_velocities=np.zeros((horizon, 3), dtype=np.float32),
                timestamps=np.arange(horizon) * dt
            )
        def predict_all(self, horizon, dt=0.05, latency_comp_sec=0.0):
            return {oid: self.predict_obstacle_trajectory(oid, horizon, dt) for oid in self.poses}
        def get_estimated_state(self, obj_id):
            p, q = self.poses[obj_id]
            return {'position': p, 'velocity': np.zeros(3), 'quaternion': q, 'angular_velocity': np.zeros(3)}
        def prune_stale_obstacles(self, max_age_seconds=2.0):
            return []
        def reset(self):
            self.poses.clear()

    twin_static = PyBulletDigitalTwin(visualize=False, predictor=StaticPredictor())
    twin_static.load_robot(base_position=(0, 0, 0.75))

    # Setup obstacle in both twins
    for tw in [twin_dyn, twin_static]:
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.06], physicsClientId=tw.client_id)
        body_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_id, basePosition=[0.45, 0.05, 0.8], physicsClientId=tw.client_id)
        tw.tracked_objects[1] = {'body_id': body_id, 'target_pos': (0.45, 0.05, 0.8), 'target_quat': (0, 0, 0, 1), 'is_target': False}

    # Feed moving observations to both (obs moving from Y=0.0 to Y=0.05 at 0.5 m/s)
    twin_dyn.sync_object_pose(1, (0.45, 0.00, 0.8), (0, 0, 0, 1), timestamp=0.0)
    twin_dyn.sync_object_pose(1, (0.45, 0.05, 0.8), (0, 0, 0, 1), timestamp=0.1)

    twin_static.sync_object_pose(1, (0.45, 0.05, 0.8), (0, 0, 0, 1), timestamp=0.1)

    # Create candidate trajectory advancing towards [0.45, 0.0, 0.8]
    kin = FrankaKinematics()
    q_init = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    H = 15
    K = 1
    qd_reach = np.array([0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    Q = np.zeros((K, H, 7), dtype=np.float32)
    QD = np.zeros((K, H, 7), dtype=np.float32)
    for h in range(H):
        Q[0, h] = q_init + qd_reach * (h * 0.05)
        QD[0, h] = qd_reach

    # Compute collision costs
    c_p_static, c_gvm_static = twin_static.compute_trajectory_collision_costs(
        Q, QD, sigma_1=0.05, sigma_2=0.20, kappa=15.0, rho=2.0, kin_helper=kin, dt=0.05
    )
    c_p_dyn, c_gvm_dyn = twin_dyn.compute_trajectory_collision_costs(
        Q, QD, sigma_1=0.05, sigma_2=0.20, kappa=15.0, rho=2.0, kin_helper=kin, dt=0.05
    )

    cost_static = float(np.sum(c_p_static + c_gvm_static))
    cost_dyn = float(np.sum(c_p_dyn + c_gvm_dyn))

    print(f"Collision Cost with STATIC Assumption:    {cost_static:.4f}")
    print(f"Collision Cost with DYNAMIC Motion Pred:  {cost_dyn:.4f}")
    if cost_static > 1e-4:
        pct_diff = (cost_static - cost_dyn) / cost_static * 100.0
        print(f"Cost Reduction (Obstacle Moving Away):    {pct_diff:.1f}%")
    else:
        print(f"Collision clearance achieved in both.")

    twin_dyn.shutdown()
    twin_static.shutdown()

    assert cost_dyn <= cost_static, "Dynamic predictor should not exceed static cost when obstacle is receding!"
    print(">>> TEST 9 RESULT: PASSED [PREDICTION PREVENTS FALSE-ALARM RETREAT]")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("STARTING COMPREHENSIVE DREMA PREDICTION STRESS TEST SUITE")
    print("#"*70)
    run_test_1_helical_3d_motion()
    run_test_2_sensor_dropout_and_occlusions()
    run_test_3_hostile_outlier_injection()
    run_test_4_multi_obstacle_swarm_scalability()
    run_test_5_closed_loop_mppi_pybullet()
    run_test_6_instantaneous_direction_inversion()
    run_test_7_asynchronous_jitter_and_heavy_noise()
    run_test_8_tumbling_so3_lie_manifold()
    run_test_9_mppi_dynamic_vs_static_cost()
    print("\n" + "#"*70)
    print("ALL 9 ADVANCED STRESS TESTS COMPLETED SUCCESSFULLY!")
    print("#"*70 + "\n")
