import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_predictor import BaseObstaclePredictor, TrajectoryPrediction


class SingleObjectKinematicFilter:
    """
    Recursive Bayesian State Estimator on SE(3) x R^6 for a single dynamic object.
    
    Translational Dynamics:
        State vector x = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]^T (9D)
        Continuous White Noise Acceleration (CWNA) / Constant Acceleration (CA) model.
        Measurement model: z = [p_x, p_y, p_z]^T.
        Equipped with Mahalanobis distance gating to reject perceptual outliers.

    Rotational Dynamics:
        State on SO(3) parameterized by unit quaternions q in H.
        Angular velocity omega in R^3 estimated via Lie algebra tangent space difference:
            omega = 2 * Log(q_t (x) q_{t-1}^*) / dt
        with exponential smoothing and forward Lie group integration.
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_quat: np.ndarray,
        initial_time: float,
        pos_meas_noise: float = 0.005,      # 5mm standard deviation on perception
        proc_acc_noise: float = 2.0,        # 2.0 m/s^2 acceleration random walk
        mahalanobis_thresh: float = 50.0,   # ~7-sigma outlier rejection threshold
        angular_smooth_alpha: float = 0.4   # Exponential filter factor on angular velocity
    ):
        self.last_timestamp = float(initial_time)
        self.pos_meas_noise = float(pos_meas_noise)
        self.proc_acc_noise = float(proc_acc_noise)
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.angular_smooth_alpha = float(angular_smooth_alpha)
        self.consecutive_rejections = 0

        # 9D State: [p(3), v(3), a(3)]
        self.x = np.zeros(9, dtype=np.float32)
        self.x[0:3] = np.array(initial_pos, dtype=np.float32)

        # Covariance P: high confidence in initial position, unobserved vel/acc
        self.P = np.eye(9, dtype=np.float32) * 1.0
        self.P[0:3, 0:3] = (self.pos_meas_noise ** 2) * np.eye(3, dtype=np.float32)
        self.P[3:6, 3:6] = 1.0 * np.eye(3, dtype=np.float32)
        self.P[6:9, 6:9] = 5.0 * np.eye(3, dtype=np.float32)

        # Measurement matrix H: maps 9D state -> 3D position
        self.H_mat = np.zeros((3, 9), dtype=np.float32)
        self.H_mat[0, 0] = self.H_mat[1, 1] = self.H_mat[2, 2] = 1.0

        # Measurement noise covariance R
        self.R_mat = (self.pos_meas_noise ** 2) * np.eye(3, dtype=np.float32)

        # Rotation state
        self.current_quat = self._normalize_quat(initial_quat)
        self.current_omega = np.zeros(3, dtype=np.float32)
        self.updates_count = 1

    @staticmethod
    def _normalize_quat(q: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        q_arr = np.array(q, dtype=np.float32)
        norm = np.linalg.norm(q_arr)
        if norm > 1e-8:
            return q_arr / norm
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _build_discrete_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs state transition matrix F(dt) and process noise covariance Q(dt)."""
        dt = max(dt, 1e-4)
        dt2 = 0.5 * dt * dt
        dt3 = dt * dt2 / 3.0
        dt4 = dt2 * dt2 / 4.0
        dt5 = dt3 * dt2 / 5.0

        F = np.eye(9, dtype=np.float32)
        for i in range(3):
            F[i, i + 3] = dt
            F[i, i + 6] = dt2
            F[i + 3, i + 6] = dt

        q_var = self.proc_acc_noise ** 2
        Q = np.zeros((9, 9), dtype=np.float32)
        for i in range(3):
            Q[i, i] = dt5 * q_var
            Q[i, i + 3] = Q[i + 3, i] = dt4 * q_var
            Q[i, i + 6] = Q[i + 6, i] = dt3 * q_var
            Q[i + 3, i + 3] = dt3 * q_var
            Q[i + 3, i + 6] = Q[i + 6, i + 3] = dt2 * q_var
            Q[i + 6, i + 6] = dt * q_var

        return F, Q

    def update(
        self,
        measured_pos: Union[np.ndarray, List[float], Tuple[float, ...]],
        measured_quat: Union[np.ndarray, List[float], Tuple[float, ...]],
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Ingests a new 6D pose observation and updates the continuous filter state.
        Returns True if observation was accepted, False if rejected as outlier.
        """
        now = time.time() if timestamp is None else float(timestamp)
        dt = now - self.last_timestamp

        # Skip backwards or zero time-step duplicate frames
        if dt < 1e-4:
            return False

        # If time gap is massive (> 15.0s), tracking was paused or reset: re-anchor state
        if dt > 15.0:
            self.x[0:3] = np.array(measured_pos, dtype=np.float32)
            self.x[3:6] = 0.0
            self.x[6:9] = 0.0
            self.P[3:6, 3:6] = 1.0 * np.eye(3, dtype=np.float32)
            self.last_timestamp = now
            return True

        # 1. Prediction Step for Translation
        F, Q = self._build_discrete_matrices(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # 2. Measurement Innovation & Mahalanobis Outlier Gating
        z = np.array(measured_pos, dtype=np.float32)
        y = z - self.H_mat @ x_pred
        S = self.H_mat @ P_pred @ self.H_mat.T + self.R_mat

        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist_sq = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            S_inv = np.eye(3, dtype=np.float32) / (self.pos_meas_noise ** 2)
            mahalanobis_dist_sq = 0.0

        # Outlier rejection: if measurement residuals are impossibly huge (glitch)
        # Includes automatic gate recovery if 3 consecutive observations are flagged (real sharp maneuver)
        if self.updates_count > 5 and mahalanobis_dist_sq > self.mahalanobis_thresh:
            self.consecutive_rejections += 1
            if self.consecutive_rejections < 3:
                self.x = x_pred
                self.P = P_pred
                self.last_timestamp = now
                return False
            # Otherwise: gate recovery after persistent maneuver
            self.consecutive_rejections = 0
            self.P[3:6, 3:6] += 0.5 * np.eye(3, dtype=np.float32)
        else:
            self.consecutive_rejections = 0

        # 3. Kalman Update
        K = P_pred @ self.H_mat.T @ S_inv
        self.x = x_pred + K @ y
        I_KH = np.eye(9, dtype=np.float32) - K @ self.H_mat
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R_mat @ K.T

        # 4. Rotation Update via SO(3) Tangent Space
        norm_meas_quat = self._normalize_quat(measured_quat)
        if np.dot(norm_meas_quat, self.current_quat) < 0.0:
            norm_meas_quat = -norm_meas_quat

        try:
            r_prev = R.from_quat(self.current_quat)
            r_meas = R.from_quat(norm_meas_quat)
            r_diff = r_meas * r_prev.inv()
            rotvec = r_diff.as_rotvec()
            meas_omega = rotvec / dt
            self.current_omega = (
                self.angular_smooth_alpha * meas_omega +
                (1.0 - self.angular_smooth_alpha) * self.current_omega
            )
        except Exception:
            pass

        self.current_quat = norm_meas_quat
        self.last_timestamp = now
        self.updates_count += 1
        return True

    def predict_forward(
        self,
        horizon_steps: int,
        dt_step: float,
        latency_comp_sec: float = 0.0
    ) -> TrajectoryPrediction:
        """
        Extrapolates state forward across H future steps using continuous kinematics.
        Optionally compensates for latency by advancing the base state prior to rollout.
        """
        H = max(1, int(horizon_steps))
        dt = float(dt_step)
        t_base = self.last_timestamp + float(latency_comp_sec)

        # Latency compensation projection
        if latency_comp_sec > 1e-4:
            F_lat, _ = self._build_discrete_matrices(latency_comp_sec)
            x0 = F_lat @ self.x
            d_theta = self.current_omega * latency_comp_sec
            if np.linalg.norm(d_theta) > 1e-6:
                r_comp = R.from_rotvec(d_theta) * R.from_quat(self.current_quat)
                q0 = r_comp.as_quat().astype(np.float32)
            else:
                q0 = self.current_quat.copy()
        else:
            x0 = self.x.copy()
            q0 = self.current_quat.copy()

        p0 = x0[0:3]
        v0 = x0[3:6]
        a0 = x0[6:9]

        time_offsets = (np.arange(H, dtype=np.float32) * dt)[:, None]

        pred_positions = p0[None, :] + time_offsets * v0[None, :] + 0.5 * (time_offsets ** 2) * a0[None, :]
        pred_velocities = v0[None, :] + time_offsets * a0[None, :]

        pred_orientations = np.zeros((H, 4), dtype=np.float32)
        r0 = R.from_quat(q0)
        omega_norm = np.linalg.norm(self.current_omega)

        if omega_norm > 1e-5:
            rotvecs = (time_offsets * self.current_omega[None, :])
            r_deltas = R.from_rotvec(rotvecs)
            r_horizon = r_deltas * r0
            pred_orientations = r_horizon.as_quat().astype(np.float32)
        else:
            pred_orientations[:] = q0[None, :]

        pred_angular_vels = np.tile(self.current_omega, (H, 1))
        pred_timestamps = t_base + np.arange(H, dtype=np.float32) * dt

        return TrajectoryPrediction(
            positions=pred_positions.astype(np.float32),
            velocities=pred_velocities.astype(np.float32),
            orientations=pred_orientations.astype(np.float32),
            angular_velocities=pred_angular_vels.astype(np.float32),
            timestamps=pred_timestamps
        )


class KalmanObstaclePredictor(BaseObstaclePredictor):
    """
    Kinematic Bayesian Multi-Obstacle Trajectory Predictor.
    Implements BaseObstaclePredictor interface using Kalman filtering on SE(3) x R^6.
    """

    def __init__(
        self,
        default_meas_noise: float = 0.005,
        default_acc_noise: float = 1.0,
        mahalanobis_thresh: float = 50.0,
        angular_smooth_alpha: float = 0.4,
        stale_timeout_sec: float = 2.0
    ):
        self.default_meas_noise = float(default_meas_noise)
        self.default_acc_noise = float(default_acc_noise)
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.angular_smooth_alpha = float(angular_smooth_alpha)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.filters: Dict[Union[int, str], SingleObjectKinematicFilter] = {}

    def update_obstacle_pose(
        self,
        obj_id: Union[int, str],
        position: Union[np.ndarray, List[float], Tuple[float, ...]],
        orientation: Union[np.ndarray, List[float], Tuple[float, ...]],
        timestamp: Optional[float] = None
    ) -> bool:
        now = time.time() if timestamp is None else float(timestamp)

        if obj_id not in self.filters:
            self.filters[obj_id] = SingleObjectKinematicFilter(
                initial_pos=np.array(position, dtype=np.float32),
                initial_quat=np.array(orientation, dtype=np.float32),
                initial_time=now,
                pos_meas_noise=self.default_meas_noise,
                proc_acc_noise=self.default_acc_noise,
                mahalanobis_thresh=self.mahalanobis_thresh,
                angular_smooth_alpha=self.angular_smooth_alpha
            )
            return True

        return self.filters[obj_id].update(position, orientation, timestamp=now)

    def predict_obstacle_trajectory(
        self,
        obj_id: Union[int, str],
        horizon: int,
        dt: float,
        latency_comp_sec: float = 0.0
    ) -> Optional[TrajectoryPrediction]:
        if obj_id not in self.filters:
            return None
        return self.filters[obj_id].predict_forward(horizon, dt, latency_comp_sec=latency_comp_sec)

    def predict_all(
        self,
        horizon: int,
        dt: float,
        latency_comp_sec: float = 0.0
    ) -> Dict[Union[int, str], TrajectoryPrediction]:
        predictions = {}
        for obj_id, flt in self.filters.items():
            predictions[obj_id] = flt.predict_forward(horizon, dt, latency_comp_sec=latency_comp_sec)
        return predictions

    def get_estimated_state(self, obj_id: Union[int, str]) -> Optional[Dict[str, np.ndarray]]:
        if obj_id not in self.filters:
            return None
        flt = self.filters[obj_id]
        return {
            'position': flt.x[0:3].copy(),
            'velocity': flt.x[3:6].copy(),
            'acceleration': flt.x[6:9].copy(),
            'orientation': flt.current_quat.copy(),
            'angular_velocity': flt.current_omega.copy(),
            'last_timestamp': flt.last_timestamp
        }

    def prune_stale_obstacles(self, timeout_sec: Optional[float] = None) -> List[Union[int, str]]:
        timeout = self.stale_timeout_sec if timeout_sec is None else float(timeout_sec)
        now = time.time()
        stale_ids = [
            oid for oid, flt in self.filters.items()
            if (now - flt.last_timestamp) > timeout
        ]
        for oid in stale_ids:
            del self.filters[oid]
        return stale_ids

    def reset(self) -> None:
        self.filters.clear()


# Backward-compatible alias
ObstacleTrajectoryPredictor = KalmanObstaclePredictor
