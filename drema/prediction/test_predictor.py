import numpy as np
import time
from drema.prediction.obstacle_predictor import ObstacleTrajectoryPredictor

def test_predictor():
    predictor = ObstacleTrajectoryPredictor()
    dt_sim = 0.033 # 30 Hz perception
    times = np.arange(0, 1.5, dt_sim)

    for t in times:
        y = 0.25 * np.cos(2.0 * np.pi * 0.5 * t)
        pos = np.array([0.4, y, 0.8], dtype=np.float32) + np.random.randn(3) * 0.001
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        predictor.update_obstacle_pose(obj_id="test_obstacle_1", position=pos, orientation=quat, timestamp=t)

    t_eval = times[-1]
    expected_vy = -0.25 * (2.0 * np.pi * 0.5) * np.sin(2.0 * np.pi * 0.5 * t_eval)
    state = predictor.get_estimated_state("test_obstacle_1")
    pred = predictor.predict_obstacle_trajectory("test_obstacle_1", horizon=20, dt=0.05)

    print("=== DREMA OBSTACLE PREDICTOR TEST ===")
    print(f"Tracked state: Pos = {np.round(state['position'], 3)}")
    print(f"Estimated Vel Y: {state['velocity'][1]:.3f} m/s (Analytical ground truth: {expected_vy:.3f} m/s)")
    print(f"Velocity estimation error: {abs(state['velocity'][1] - expected_vy):.4f} m/s")
    print(f"Trajectory forecast shape: {pred.positions.shape}")
    print(f"Predicted Y trajectory (first 5 steps): {np.round(pred.positions[:5, 1], 3)}")

    # Benchmark latency
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        predictor.update_obstacle_pose("test_obstacle_1", [0.4, 0.1, 0.8], [0, 0, 0, 1], timestamp=2.0)
    t_update = (time.perf_counter() - t0) / N * 1000.0

    t0 = time.perf_counter()
    for _ in range(N):
        _ = predictor.predict_obstacle_trajectory("test_obstacle_1", horizon=20, dt=0.05)
    t_pred = (time.perf_counter() - t0) / N * 1000.0

    print(f"Benchmark Update Latency: {t_update*1000:.1f} us ({t_update:.4f} ms)")
    print(f"Benchmark Predict Latency (H=20): {t_pred*1000:.1f} us ({t_pred:.4f} ms)")
    print(f"Total Overhead per control cycle: {(t_update + t_pred)*1000:.1f} us ({(t_update + t_pred):.4f} ms)")

if __name__ == "__main__":
    test_predictor()
