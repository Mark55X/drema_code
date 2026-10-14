import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_predictor import BaseObstaclePredictor, TrajectoryPrediction


class SingleObjectKinematicFilter:
    """
    Recursive Bayesian State Estimator on SE(3) x R^6 for a single dynamic object.

    Translational Dynamics:
        State vector x = [p_x, p_y, p_z, v_x, v_y, v_z]^T (6D)
        Constant Velocity model driven by continuous white-noise acceleration.
        Measurement model: z = [p_x, p_y, p_z]^T.
        Equipped with Mahalanobis distance gating to reject perceptual outliers.

    Rotational Dynamics:
        Multiplicative (error-state) Kalman filter on SO(3) with a constant angular velocity model:
        filtered orientation q, error state [delta_phi, omega] in the tangent space, measurement
        residual Log(q_meas q^-1).

    Static-object detection:
        The reported linear / angular velocity (used for prediction) is the filtered one only when
        it is statistically distinguishable from zero (v^T P_vv^-1 v above a chi-square threshold)
        and above a minimum magnitude; otherwise it is reported as zero. Without this, perception
        jitter on a static object is extrapolated into spurious motion over the whole MPC horizon.
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_quat: np.ndarray,
        initial_time: float,
        pos_meas_noise: float = 0.005,          # 5mm standard deviation on perception
        proc_acc_noise: float = 0.02,           # nominal white-noise acceleration intensity [m/s^2 / sqrt(Hz)]
        maneuver_acc_noise: float = 2.0,        # upper bound of the adapted intensity during maneuvers
        nis_smoothing: float = 0.5,             # EMA factor of the normalized innovation squared
        noise_adaptation_gain: float = 0.3,     # log-variance step per unit of log(NIS / 3)
        mahalanobis_thresh: float = 50.0,       # ~7-sigma outlier rejection threshold
        rot_meas_noise: float = 0.05,           # [rad] standard deviation of the perceived orientation
        proc_ang_acc_noise: float = 0.05,       # white-noise angular acceleration intensity [rad/s^2 / sqrt(Hz)]
        static_chi2_thresh: float = 11.34,      # chi-square 3 DoF, 99%: velocity not distinguishable from zero
        min_moving_speed: float = 0.02,         # [m/s] slower objects are reported as static
        min_moving_omega: float = 0.15          # [rad/s] slower rotations are reported as zero
    ):
        """
        The process noise is adapted from the innovation: a consistent filter has a normalized innovation
        squared (NIS) of 3 on average (3 DoF). The variance scale is integrated in log space towards
        NIS = 3 and clipped to [proc_acc_noise^2, maneuver_acc_noise^2]; a proportional scale would
        collapse back as soon as the raised noise made the filter consistent again. A low nominal
        intensity keeps static and constant-velocity objects stable under perception jitter, while
        sustained maneuvers raise the gain within a few frames.
        """
        self.pos_meas_noise = float(pos_meas_noise)
        self.proc_acc_noise = float(proc_acc_noise)
        self.maneuver_acc_noise = max(float(maneuver_acc_noise), self.proc_acc_noise)
        self.nis_smoothing = float(nis_smoothing)
        self.noise_adaptation_gain = float(noise_adaptation_gain)
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.rot_meas_noise = float(rot_meas_noise)
        self.proc_ang_acc_noise = float(proc_ang_acc_noise)
        self.static_chi2_thresh = float(static_chi2_thresh)
        self.min_moving_speed = float(min_moving_speed)
        self.min_moving_omega = float(min_moving_omega)

        # Measurement matrix H: maps 6D state -> 3D position (same structure for the rotation error state)
        self.H_mat = np.zeros((3, 6), dtype=np.float64)
        self.H_mat[0, 0] = self.H_mat[1, 1] = self.H_mat[2, 2] = 1.0

        self.R_mat = (self.pos_meas_noise ** 2) * np.eye(3, dtype=np.float64)
        self.R_rot = (self.rot_meas_noise ** 2) * np.eye(3, dtype=np.float64)

        self._anchor(initial_pos, initial_quat, initial_time)

    def _anchor(self, position, quat, timestamp: float) -> None:
        """(Re)initializes the state at a measured pose with unknown velocity."""
        self.last_timestamp = float(timestamp)
        self.consecutive_rejections = 0
        self.x = np.zeros(6, dtype=np.float64)
        self.x[0:3] = np.asarray(position, dtype=np.float64)
        self.P = np.zeros((6, 6), dtype=np.float64)
        self.P[0:3, 0:3] = (self.pos_meas_noise ** 2) * np.eye(3)
        self.P[3:6, 3:6] = 1.0 * np.eye(3)
        self.current_quat = self._normalize_quat(quat)
        self.current_omega = np.zeros(3, dtype=np.float64)
        self.P_rot = np.zeros((6, 6), dtype=np.float64)
        self.P_rot[0:3, 0:3] = self.R_rot.copy()
        self.P_rot[3:6, 3:6] = 1.0 * np.eye(3)
        self.updates_count = 1
        self.nis_ema = 3.0
        self.log_variance_scale = 0.0

    @property
    def adapted_acc_noise(self) -> float:
        return self.proc_acc_noise * float(np.exp(0.5 * self.log_variance_scale))

    def _adapt_process_noise(self, nis: float) -> None:
        self.nis_ema = (1.0 - self.nis_smoothing) * self.nis_ema + self.nis_smoothing * nis
        max_log_scale = 2.0 * np.log(self.maneuver_acc_noise / self.proc_acc_noise)
        self.log_variance_scale = float(np.clip(
            self.log_variance_scale + self.noise_adaptation_gain * np.log(max(self.nis_ema, 1e-3) / 3.0), 0.0, max_log_scale
        ))

    @staticmethod
    def _normalize_quat(q: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        q_arr = np.array(q, dtype=np.float64)
        norm = np.linalg.norm(q_arr)
        if norm > 1e-8:
            return q_arr / norm
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _build_discrete_matrices(dt: float, acc_noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-velocity transition F(dt) and process noise Q(dt) for a [value, rate] state in 3D."""
        dt = max(dt, 1e-4)
        F = np.eye(6, dtype=np.float64)
        F[0:3, 3:6] = dt * np.eye(3)

        q_var = acc_noise ** 2
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = (dt ** 3 / 3.0) * q_var * np.eye(3)
        Q[0:3, 3:6] = Q[3:6, 0:3] = (dt ** 2 / 2.0) * q_var * np.eye(3)
        Q[3:6, 3:6] = dt * q_var * np.eye(3)
        return F, Q

    def is_moving(self) -> bool:
        """True when the filtered velocity is statistically and physically distinguishable from zero."""
        v = self.x[3:6]
        if float(np.linalg.norm(v)) < self.min_moving_speed:
            return False
        try:
            significance = float(v @ np.linalg.solve(self.P[3:6, 3:6], v))
        except np.linalg.LinAlgError:
            return True
        return significance > self.static_chi2_thresh

    def effective_velocity(self) -> np.ndarray:
        return self.x[3:6].copy() if self.is_moving() else np.zeros(3, dtype=np.float64)

    def is_rotating(self) -> bool:
        """True when the filtered angular velocity is statistically and physically distinguishable from zero."""
        w = self.current_omega
        if float(np.linalg.norm(w)) < self.min_moving_omega:
            return False
        try:
            significance = float(w @ np.linalg.solve(self.P_rot[3:6, 3:6], w))
        except np.linalg.LinAlgError:
            return True
        return significance > self.static_chi2_thresh

    def effective_omega(self) -> np.ndarray:
        return self.current_omega.copy() if self.is_rotating() else np.zeros(3, dtype=np.float64)

    def _update_rotation(self, measured_quat, dt: float) -> None:
        """Error-state update: predict q <- Exp(omega dt) q, correct with the residual Log(q_meas q^-1)."""
        F, Q = self._build_discrete_matrices(dt, self.proc_ang_acc_noise)
        q_pred = R.from_rotvec(self.current_omega * dt) * R.from_quat(self.current_quat)
        P_pred = F @ self.P_rot @ F.T + Q

        residual = (R.from_quat(self._normalize_quat(measured_quat)) * q_pred.inv()).as_rotvec()
        S = self.H_mat @ P_pred @ self.H_mat.T + self.R_rot
        K = P_pred @ self.H_mat.T @ np.linalg.inv(S)
        delta = K @ residual
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H_mat
        self.P_rot = I_KH @ P_pred @ I_KH.T + K @ self.R_rot @ K.T

        self.current_quat = (R.from_rotvec(delta[0:3]) * q_pred).as_quat()
        self.current_omega = self.current_omega + delta[3:6]

    def update(
        self,
        measured_pos: Union[np.ndarray, List[float], Tuple[float, ...]],
        measured_quat: Union[np.ndarray, List[float], Tuple[float, ...]],
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Ingests a new 6D pose observation and updates the continuous filter state.
        Returns True if observation was accepted, False if rejected as outlier or duplicate.
        """
        now = time.time() if timestamp is None else float(timestamp)
        dt = now - self.last_timestamp

        # The observation clock went backwards (e.g. simulation restarted without a reset):
        # the previous state belongs to another timeline, so re-anchor instead of rejecting forever.
        if dt < -1e-3:
            self._anchor(measured_pos, measured_quat, now)
            return True

        # Skip zero time-step duplicate frames
        if dt < 1e-4:
            return False

        # If time gap is massive (> 15.0s), tracking was paused or reset: re-anchor state
        if dt > 15.0:
            self._anchor(measured_pos, measured_quat, now)
            return True

        # 1. Prediction Step for Translation
        F, Q = self._build_discrete_matrices(dt, self.adapted_acc_noise)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # 2. Measurement Innovation & Mahalanobis Outlier Gating
        z = np.array(measured_pos, dtype=np.float64)
        y = z - self.H_mat @ x_pred
        S = self.H_mat @ P_pred @ self.H_mat.T + self.R_mat

        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist_sq = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            S_inv = np.eye(3, dtype=np.float64) / (self.pos_meas_noise ** 2)
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
            P_pred[3:6, 3:6] += 0.5 * np.eye(3, dtype=np.float64)
            S = self.H_mat @ P_pred @ self.H_mat.T + self.R_mat
            S_inv = np.linalg.inv(S)
        else:
            self.consecutive_rejections = 0

        # Rejected glitches never reach this point, so isolated outliers do not inflate the process noise
        self._adapt_process_noise(min(mahalanobis_dist_sq, self.mahalanobis_thresh))

        # 3. Kalman Update (Joseph form)
        K = P_pred @ self.H_mat.T @ S_inv
        self.x = x_pred + K @ y
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H_mat
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R_mat @ K.T

        # 4. Rotation Update on SO(3)
        self._update_rotation(measured_quat, dt)
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
        Extrapolates state forward across H future steps with constant (effective) velocity.
        Optionally compensates for latency by advancing the base state prior to rollout.
        """
        H = max(1, int(horizon_steps))
        dt = float(dt_step)
        t_base = self.last_timestamp + float(latency_comp_sec)
        v0 = self.effective_velocity()
        omega = self.effective_omega()

        p0 = self.x[0:3] + v0 * float(latency_comp_sec)
        if latency_comp_sec > 1e-4 and np.linalg.norm(omega) > 0.0:
            q0 = (R.from_rotvec(omega * latency_comp_sec) * R.from_quat(self.current_quat)).as_quat()
        else:
            q0 = self.current_quat.copy()

        time_offsets = (np.arange(H, dtype=np.float64) * dt)[:, None]
        pred_positions = p0[None, :] + time_offsets * v0[None, :]
        pred_velocities = np.tile(v0, (H, 1))

        if np.linalg.norm(omega) > 0.0:
            pred_orientations = (R.from_rotvec(time_offsets * omega[None, :]) * R.from_quat(q0)).as_quat()
        else:
            pred_orientations = np.tile(q0, (H, 1))

        return TrajectoryPrediction(
            positions=pred_positions.astype(np.float32),
            velocities=pred_velocities.astype(np.float32),
            orientations=pred_orientations.astype(np.float32),
            angular_velocities=np.tile(omega, (H, 1)).astype(np.float32),
            timestamps=t_base + np.arange(H, dtype=np.float64) * dt
        )


class KalmanObstaclePredictor(BaseObstaclePredictor):
    """
    Kinematic Bayesian Multi-Obstacle Trajectory Predictor.
    Implements BaseObstaclePredictor interface using Kalman filtering on SE(3) x R^6.
    """

    def __init__(
        self,
        default_meas_noise: float = 0.005,
        default_acc_noise: float = 0.02,
        maneuver_acc_noise: float = 2.0,
        mahalanobis_thresh: float = 50.0,
        rot_meas_noise: float = 0.05,
        proc_ang_acc_noise: float = 0.05,
        stale_timeout_sec: float = 2.0,
        static_chi2_thresh: float = 11.34,
        min_moving_speed: float = 0.02,
        min_moving_omega: float = 0.15
    ):
        self.default_meas_noise = float(default_meas_noise)
        self.default_acc_noise = float(default_acc_noise)
        self.maneuver_acc_noise = float(maneuver_acc_noise)
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.rot_meas_noise = float(rot_meas_noise)
        self.proc_ang_acc_noise = float(proc_ang_acc_noise)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.static_chi2_thresh = float(static_chi2_thresh)
        self.min_moving_speed = float(min_moving_speed)
        self.min_moving_omega = float(min_moving_omega)
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
                initial_pos=np.array(position, dtype=np.float64),
                initial_quat=np.array(orientation, dtype=np.float64),
                initial_time=now,
                pos_meas_noise=self.default_meas_noise,
                proc_acc_noise=self.default_acc_noise,
                maneuver_acc_noise=self.maneuver_acc_noise,
                mahalanobis_thresh=self.mahalanobis_thresh,
                rot_meas_noise=self.rot_meas_noise,
                proc_ang_acc_noise=self.proc_ang_acc_noise,
                static_chi2_thresh=self.static_chi2_thresh,
                min_moving_speed=self.min_moving_speed,
                min_moving_omega=self.min_moving_omega
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
        """
        'velocity' and 'angular_velocity' are the effective values used for prediction
        (zero for a static object); 'raw_velocity' is the unthresholded filter estimate.
        'acceleration' is always zero under the constant-velocity model.
        """
        if obj_id not in self.filters:
            return None
        flt = self.filters[obj_id]
        return {
            'position': flt.x[0:3].copy(),
            'velocity': flt.effective_velocity(),
            'raw_velocity': flt.x[3:6].copy(),
            'acceleration': np.zeros(3, dtype=np.float64),
            'orientation': flt.current_quat.copy(),
            'angular_velocity': flt.effective_omega(),
            'raw_angular_velocity': flt.current_omega.copy(),
            'is_static': not flt.is_moving(),
            'is_rotating': flt.is_rotating(),
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
