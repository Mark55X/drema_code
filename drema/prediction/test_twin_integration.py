import time
import numpy as np
import pybullet as p
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.controller.franka_kinematics import FrankaKinematics
from drema.prediction import BaseObstaclePredictor, TrajectoryPrediction

def test_twin_integration():
    print("=== Testing PyBullet Digital Twin + Generic Predictor Integration ===")
    twin = PyBulletDigitalTwin(visualize=False)
    twin.load_robot(base_position=(0, 0, 0.75))
    kin = FrankaKinematics(base_position=np.array([0, 0, 0.75]))

    # 1. Spawn a generic dynamic obstacle (e.g. ID 101)
    obj_id = 101
    box_half_ext = (0.05, 0.05, 0.05)
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_ext, physicsClientId=twin.client_id)
    body_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_id, basePosition=[0.4, 0.0, 0.8], physicsClientId=twin.client_id)
    twin.tracked_objects[obj_id] = {
        'body_id': body_id,
        'target_pos': (0.4, 0.0, 0.8),
        'target_quat': (0.0, 0.0, 0.0, 1.0),
        'is_target': False
    }

    # 2. Simulate incoming tracking poses with moving velocity vy = 0.2 m/s
    t_start = time.time()
    for step in range(5):
        t = t_start + step * 0.033
        pos = (0.4, 0.0 + step * 0.033 * 0.2, 0.8)
        quat = (0.0, 0.0, 0.0, 1.0)
        twin.sync_object_pose(obj_id, pos, quat, timestamp=t)

    # Verify that predictor recorded state and estimated velocity
    state = twin.predictor.get_estimated_state(obj_id)
    print(f"Object {obj_id} filtered state:")
    print(f"  Position: {np.round(state['position'], 4)}")
    print(f"  Velocity: {np.round(state['velocity'], 4)} m/s")

    # 3. Evaluate candidate trajectories collision costs
    K = 10
    H = 15
    q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    Q = np.tile(q_home, (K, H, 1))
    QD = np.zeros((K, H, 7), dtype=np.float32)

    t0 = time.perf_counter()
    coll_p, coll_gvm = twin.compute_trajectory_collision_costs(
        Q=Q,
        QD=QD,
        sigma_1=0.03,
        sigma_2=0.20,
        kappa=10.0,
        rho=0.8,
        kin_helper=kin,
        dt=0.05
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"compute_trajectory_collision_costs executed in {elapsed_ms:.2f} ms")
    print(f"coll_p shape: {coll_p.shape}, max: {coll_p.max():.4f}")
    print(f"coll_gvm shape: {coll_gvm.shape}, max: {coll_gvm.max():.4f}")

    # Verify obstacle was restored to t0 pose
    cur_pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=twin.client_id)
    print(f"Restored t0 pose: {np.round(cur_pos, 4)}")
    assert np.allclose(cur_pos, twin.tracked_objects[obj_id]['target_pos'], atol=1e-3), "Obstacle pose not restored to t0!"

    # 4. Pluggability test: pass a custom user dummy predictor implementing BaseObstaclePredictor
    class CustomDummyPredictor(BaseObstaclePredictor):
        def update_obstacle_pose(self, obj_id, position, orientation, timestamp=None):
            return True
        def predict_obstacle_trajectory(self, obj_id, horizon, dt, latency_comp_sec=0.0):
            return None
        def predict_all(self, horizon, dt, latency_comp_sec=0.0):
            # Returns dummy zero positions
            return {
                obj_id: TrajectoryPrediction(
                    positions=np.zeros((horizon, 3), dtype=np.float32),
                    velocities=np.zeros((horizon, 3), dtype=np.float32),
                    orientations=np.zeros((horizon, 4), dtype=np.float32),
                    angular_velocities=np.zeros((horizon, 3), dtype=np.float32),
                    timestamps=np.zeros(horizon, dtype=np.float32)
                )
            }
        def get_estimated_state(self, obj_id):
            return None
        def reset(self):
            pass

    twin.predictor = CustomDummyPredictor()
    coll_p_dummy, coll_gvm_dummy = twin.compute_trajectory_collision_costs(
        Q=Q, QD=QD, sigma_1=0.03, sigma_2=0.20, kappa=10.0, rho=0.8, kin_helper=kin, dt=0.05
    )
    print("Pluggability test with CustomDummyPredictor: SUCCESS!")

    twin.shutdown()
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    test_twin_integration()
