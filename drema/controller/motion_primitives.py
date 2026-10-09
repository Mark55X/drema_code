#!/usr/bin/env python
"""
Motion Primitive Library for Franka Panda Manipulator (MP-PMPPI).

Synthesizes concepts from:
- Mathisen et al. (MP-MPPI, 2026), Section 2.3: Ingestion of structured Motion Primitives into MPPI.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .franka_kinematics import FrankaKinematics


@dataclass
class _CartesianPrimitive:
    """
    A point attached to a robot link that travels along a planned path:
    a straight line (arc_height = 0) or a parabolic arc bulging along arc_normal.
    speed = 0 holds the point in place (pure reorientation).
    """
    name: str
    link_idx: int
    local_point: np.ndarray
    direction: np.ndarray
    speed: float
    angular_velocity: Optional[np.ndarray] = None
    arc_normal: Optional[np.ndarray] = None
    arc_height: float = 0.0
    arc_length: float = 0.0


class MotionPrimitiveLibrary:
    """
    Generates and manages task-specific and exploratory motion primitives (U_p)
    in the joint acceleration space over lookahead horizon H.
    """

    def __init__(
        self,
        kinematics: FrankaKinematics,
        horizon: int = 20,
        dt: float = 0.05,
        max_joint_acc: float = 0.5,
        rot_tracking_gain: float = 2.0,
        align_rot_gain: float = 3.0,
        align_rot_min_error: float = 0.02,
        approach_vel_slow: float = 0.15,
        approach_vel_fast: float = 0.30,
        arc_height: float = 0.06,
        evade_min_distance: float = 0.01,
        evade_max_distance: float = 0.35,
        evade_ttc_threshold: float = 2.0,
        evade_min_speed: float = 0.03,
        evade_imminent_distance: float = 0.06,
        evade_retreat_speed: float = 0.15,
        evade_lift_speed: float = 0.12,
        descend_speed: float = 0.08,
        lift_speed: float = 0.10,
        lateral_speed: float = 0.12,
        path_feedback_gain: float = 4.0,
        nullspace_gain: float = 0.3
    ):
        """
        :param kinematics: FrankaKinematics instance for Jacobian and FK computations.
        :param horizon: Prediction horizon H steps.
        :param dt: Timestep delta (seconds).
        :param max_joint_acc: Maximum joint acceleration limit [rad/s^2].
        :param rot_tracking_gain: Proportional gain converting orientation error to angular twist [rad/s per rad].
        :param align_rot_gain: Proportional gain for dedicated orientation alignment primitive [rad/s per rad].
        :param align_rot_min_error: Orientation error [rad] below which align_rot is not generated
                                    (it would coincide with brake_hold).
        :param approach_vel_slow: Speed limit for slow approach primitive [m/s].
        :param approach_vel_fast: Speed limit for fast approach primitive [m/s].
        :param arc_height: Peak height [m] of the arc_to_target bulge above the straight approach line.
        :param evade_min_distance: Minimum centroid distance for the legacy centroid-based evasion input [m].
        :param evade_max_distance: Maximum surface distance considered for dynamic evasion [m].
        :param evade_ttc_threshold: Time-To-Collision threshold to trigger active evasion [s].
        :param evade_min_speed: Minimum speed threshold to classify obstacle as dynamic [m/s].
        :param evade_imminent_distance: Immediate physical proximity margin triggering evasion [m].
        :param evade_retreat_speed: Linear retreat velocity magnitude for evasion [m/s].
        :param evade_lift_speed: Upward vertical lift velocity component for evade_over [m/s].
        :param descend_speed: Speed limit for vertical descent primitive [m/s].
        :param lift_speed: Speed limit for vertical retract/lift primitive [m/s].
        :param lateral_speed: Speed limit for sideways evasion primitives [m/s].
        :param path_feedback_gain: Gain [1/s] pulling the controlled point back onto its planned path.
        :param nullspace_gain: Gain [1/s] of the null-space posture attraction towards Q_REST.
        """
        self.kin = kinematics
        self.H = horizon
        self.dt = dt
        self.max_acc = max_joint_acc
        self.rot_tracking_gain = rot_tracking_gain
        self.align_rot_gain = align_rot_gain
        self.align_rot_min_error = float(align_rot_min_error)
        self.approach_vel_slow = approach_vel_slow
        self.approach_vel_fast = approach_vel_fast
        self.arc_height = float(arc_height)
        self.evade_min_distance = float(evade_min_distance)
        self.evade_max_distance = float(evade_max_distance)
        self.evade_ttc_threshold = float(evade_ttc_threshold)
        self.evade_min_speed = float(evade_min_speed)
        self.evade_imminent_distance = float(evade_imminent_distance)
        self.evade_retreat_speed = float(evade_retreat_speed)
        self.evade_lift_speed = float(evade_lift_speed)
        self.descend_speed = float(descend_speed)
        self.lift_speed = float(lift_speed)
        self.lateral_speed = float(lateral_speed)
        self.path_feedback_gain = float(path_feedback_gain)
        self.nullspace_gain = float(nullspace_gain)
        self.last_primitive_names: List[str] = []
        self.last_evasion_threat: Optional[dict] = None

    def generate_primitives(
        self,
        q_current: np.ndarray,
        qd_current: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
        target_rot: Optional[np.ndarray] = None,
        obstacles: Optional[List[dict]] = None,
        proximity: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        Generates the full library of N_p motion primitives:
        U_p = [u_{mp, 1}, u_{mp, 2}, ..., u_{mp, N_p}] in R^{N_p x H x 7}.

        :param q_current: Current robot joint angles (shape: [7]).
        :param qd_current: Current robot joint velocities (shape: [7]).
        :param target_pos: Optional 3D position [x, y, z] of target goal in world frame.
        :param target_rot: Optional 3x3 target rotation matrix.
        :param obstacles: Legacy input: obstacles as {'position', 'velocity'} centroids, evaluated
                          against the end-effector only. Ignored when `proximity` is given.
        :param proximity: Closest robot/obstacle surface points from the Digital Twin
                          (see BaseDigitalTwin.get_obstacle_proximity).
        :return: Array of shape [N_p, H, 7] containing joint acceleration control sequences.
        """
        q_current = np.asarray(q_current, dtype=np.float64)
        qd_current = np.asarray(qd_current, dtype=np.float64)
        cartesian: List[_CartesianPrimitive] = []
        ee_pos, ee_rot = self.kin.forward_kinematics_ee(q_current)

        def ee_line(name, velocity, angular_velocity=None):
            speed = float(np.linalg.norm(velocity))
            direction = velocity / speed if speed > 1e-9 else np.zeros(3)
            return _CartesianPrimitive(name, 7, self.kin.EE_OFFSET.astype(np.float64), direction, speed, angular_velocity)

        rot_err = None
        if target_rot is not None:
            rot_err = 0.5 * (
                np.cross(ee_rot[:, 0], target_rot[:, 0]) +
                np.cross(ee_rot[:, 1], target_rot[:, 1]) +
                np.cross(ee_rot[:, 2], target_rot[:, 2])
            )

        # ---------------------------------------------------------------------
        # 1. 6-DoF Cartesian Approach Primitives (Target Seeking + Reorientation)
        # ---------------------------------------------------------------------
        if target_pos is not None:
            dir_to_goal = target_pos - ee_pos
            dist_to_goal = float(np.linalg.norm(dir_to_goal))
            if dist_to_goal > 1e-4:
                unit_dir = dir_to_goal / dist_to_goal
                v_slow = unit_dir * min(self.approach_vel_slow, dist_to_goal / (self.H * self.dt))
                v_fast = unit_dir * min(self.approach_vel_fast, dist_to_goal / (self.H * self.dt * 0.5))
                w_des = rot_err * self.rot_tracking_gain if rot_err is not None else None
                cartesian.append(ee_line("appr_slow", v_slow, w_des))
                cartesian.append(ee_line("appr_fast", v_fast, w_des))

                # Arc towards the target: the chord is the whole segment to the target, so at each
                # replanning the arm rises first and the bulge shrinks as the target gets closer.
                up = np.array([0.0, 0.0, 1.0])
                normal = up - np.dot(up, unit_dir) * unit_dir
                if np.linalg.norm(normal) > 0.3:
                    arc = ee_line("arc_to_target", v_slow, w_des)
                    arc.arc_normal = normal / np.linalg.norm(normal)
                    arc.arc_height = min(self.arc_height, 0.5 * dist_to_goal)
                    arc.arc_length = dist_to_goal
                    cartesian.append(arc)

        # ---------------------------------------------------------------------
        # 2. Vertical Insertion & Retraction (Bottleneck & Peg-in-Hole)
        # ---------------------------------------------------------------------
        cartesian.append(ee_line("descend_z", np.array([0.0, 0.0, -self.descend_speed])))
        cartesian.append(ee_line("lift_z", np.array([0.0, 0.0, self.lift_speed])))

        # ---------------------------------------------------------------------
        # 3. Lateral & Reactive Obstacle Evasion Primitives (Bypass Obstacles)
        # ---------------------------------------------------------------------
        cartesian.append(ee_line("side_left", np.array([0.0, self.lateral_speed, 0.0])))
        cartesian.append(ee_line("side_right", np.array([0.0, -self.lateral_speed, 0.0])))

        if proximity is None and obstacles:
            proximity = self._centroid_proximity(ee_pos, obstacles)
        threat = self._most_exposed_point(q_current, qd_current, proximity or [])
        self.last_evasion_threat = threat
        if threat is not None:
            normal = threat['normal']
            v_over = normal * (self.evade_retreat_speed * 0.5) + np.array([0.0, 0.0, self.evade_lift_speed])
            for name, velocity in (("evade_obs", normal * self.evade_retreat_speed), ("evade_over", v_over)):
                speed = float(np.linalg.norm(velocity))
                cartesian.append(_CartesianPrimitive(
                    name, threat['link_idx'], threat['local_point'], velocity / speed, speed
                ))

        # ---------------------------------------------------------------------
        # 4. Pure Rotational Primitive (Orientation Alignment about the TCP)
        # ---------------------------------------------------------------------
        if rot_err is not None and np.linalg.norm(rot_err) > self.align_rot_min_error:
            cartesian.append(ee_line("align_rot", np.zeros(3), rot_err * self.align_rot_gain))

        U_cart = self._integrate_cartesian_primitives(q_current, qd_current, cartesian)

        # ---------------------------------------------------------------------
        # 5. Deceleration / Brake Primitive (Safe Stop / Hold)
        # ---------------------------------------------------------------------
        U_brake = self._create_braking_primitive(qd_current)[None]

        self.last_primitive_names = [p.name for p in cartesian] + ["brake_hold"]
        U_p = np.concatenate([U_cart, U_brake], axis=0).astype(np.float32)
        return np.clip(U_p, -self.max_acc, self.max_acc)

    def _centroid_proximity(self, ee_pos: np.ndarray, obstacles: List[dict]) -> List[dict]:
        """Adapts legacy centroid obstacles to proximity entries on the end-effector."""
        entries = []
        for obs in obstacles:
            vec_away = ee_pos - np.asarray(obs.get('position', (0.0, 0.0, 0.0)), dtype=np.float64)
            dist = float(np.linalg.norm(vec_away))
            if dist <= self.evade_min_distance:
                continue
            entries.append({
                'distance': dist,
                'point_world': ee_pos,
                'normal': vec_away / dist,
                'link_idx': 7,
                'local_point': self.kin.EE_OFFSET.astype(np.float64),
                'velocity': obs.get('velocity', (0.0, 0.0, 0.0))
            })
        return entries

    def _most_exposed_point(self, q: np.ndarray, qd: np.ndarray, proximity: List[dict]) -> Optional[dict]:
        """
        Selects the robot surface point under the most urgent threat: imminent contacts first
        (closest wins), then approaching obstacles by time-to-collision. Closing speed uses the
        relative velocity of the robot point (J(q) qd) and of the obstacle along the surface normal.
        """
        if not proximity:
            return None

        link_idx = np.array([int(e['link_idx']) for e in proximity], dtype=np.int64)
        local_points = np.array([
            e['local_point'] if e.get('local_point') is not None
            else self.kin.world_to_link_point(q, int(e['link_idx']), e['point_world'])
            for e in proximity
        ], dtype=np.float64)
        J, _ = self.kin.batch_point_jacobian(np.tile(q, (len(proximity), 1)), link_idx, local_points)
        point_vel = J[:, :3, :] @ qd

        best, best_key = None, None
        for i, entry in enumerate(proximity):
            dist = float(entry['distance'])
            if dist >= self.evade_max_distance:
                continue
            normal = np.asarray(entry['normal'], dtype=np.float64)
            normal = normal / max(np.linalg.norm(normal), 1e-9)
            obs_vel = np.asarray(entry.get('velocity', (0.0, 0.0, 0.0)), dtype=np.float64)
            closing_speed = float(-np.dot(point_vel[i] - obs_vel, normal))
            is_dynamic = float(np.linalg.norm(obs_vel)) > self.evade_min_speed
            ttc = max(dist, 0.0) / closing_speed if closing_speed > 0.02 else float('inf')

            if dist < self.evade_imminent_distance:
                key = (0, dist)
            elif is_dynamic and ttc < self.evade_ttc_threshold:
                key = (1, ttc)
            else:
                continue
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    'link_idx': int(link_idx[i]),
                    'local_point': local_points[i],
                    'normal': normal,
                    'distance': dist,
                    'ttc': ttc,
                    'obj_id': entry.get('obj_id')
                }
        return best

    def _integrate_cartesian_primitives(
        self,
        q_init: np.ndarray,
        qd_init: np.ndarray,
        prims: List[_CartesianPrimitive]
    ) -> np.ndarray:
        """
        Converts all Cartesian primitives into joint acceleration sequences in one batched
        forward integration over H steps. At each step:
        - the desired point velocity is the path tangent times the speed, plus a feedback term
          pulling the point back onto its path (cross-track error only, so lagging behind the
          nominal timing does not cause a catch-up surge);
        - joint velocities come from the damped least-squares pseudo-inverse, plus a null-space
          attraction towards Q_REST that leaves the task velocity unchanged;
        - the task acceleration is scaled uniformly to the limit, preserving its direction; the
          null-space posture only uses the acceleration budget the task leaves free, so it never
          slows the task down.

        :return: Joint acceleration sequences [P, H, 7].
        """
        P = len(prims)
        u_seqs = np.zeros((P, self.H, 7), dtype=np.float32)
        if P == 0:
            return u_seqs

        link_idx = np.array([p.link_idx for p in prims], dtype=np.int64)
        local_points = np.array([p.local_point for p in prims], dtype=np.float64)
        direction = np.array([p.direction for p in prims], dtype=np.float64)
        speed = np.array([p.speed for p in prims], dtype=np.float64)
        use_angular = np.array([p.angular_velocity is not None for p in prims], dtype=bool)
        angular = np.array([p.angular_velocity if p.angular_velocity is not None else np.zeros(3) for p in prims], dtype=np.float64)
        arc_normal = np.array([p.arc_normal if p.arc_normal is not None else np.zeros(3) for p in prims], dtype=np.float64)
        arc_height = np.array([p.arc_height for p in prims], dtype=np.float64)
        arc_length = np.array([max(p.arc_length, 1e-6) for p in prims], dtype=np.float64)
        moving = speed > 1e-9

        q = np.tile(np.asarray(q_init, dtype=np.float64), (P, 1))
        qd = np.tile(np.asarray(qd_init, dtype=np.float64), (P, 1))
        q_min = self.kin.Q_MIN.astype(np.float64)
        q_max = self.kin.Q_MAX.astype(np.float64)
        q_rest = self.kin.Q_REST.astype(np.float64)
        _, p_start = self.kin.batch_point_jacobian(q, link_idx, local_points)
        eye_7 = np.eye(7)

        for h in range(self.H):
            J, p = self.kin.batch_point_jacobian(q, link_idx, local_points)

            progress = np.sum((p - p_start) * direction, axis=1)
            s = np.clip(progress / arc_length, 0.0, 1.0)
            on_arc = (progress > 0.0) & (progress < arc_length)
            offset = 4.0 * arc_height * s * (1.0 - s)
            slope = np.where(on_arc, 4.0 * arc_height * (1.0 - 2.0 * s) / arc_length, 0.0)
            tangent = direction + arc_normal * slope[:, None]
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)

            path_error = p_start + direction * progress[:, None] + arc_normal * offset[:, None] - p
            cross_track = path_error - np.sum(path_error * tangent, axis=1, keepdims=True) * tangent
            v_linear = np.where(
                moving[:, None],
                speed[:, None] * tangent + self.path_feedback_gain * cross_track,
                self.path_feedback_gain * (p_start - p)
            )

            qd_task = np.zeros((P, 7), dtype=np.float64)
            qd_posture = np.zeros((P, 7), dtype=np.float64)
            for mask, rows, damping in ((~use_angular, 3, 0.01), (use_angular, 6, 0.02)):
                if not np.any(mask):
                    continue
                J_task = J[mask, :rows, :]
                twist = np.concatenate([v_linear[mask], angular[mask]], axis=1)[:, :rows]
                J_pinv = J_task.transpose(0, 2, 1) @ np.linalg.inv(J_task @ J_task.transpose(0, 2, 1) + damping * np.eye(rows))
                nullspace = eye_7 - J_pinv @ J_task
                posture = self.nullspace_gain * (q_rest - q[mask])
                qd_task[mask] = (J_pinv @ twist[..., None])[..., 0]
                qd_posture[mask] = (nullspace @ posture[..., None])[..., 0]

            qdd_task = (qd_task - qd) / self.dt
            peak = np.max(np.abs(qdd_task), axis=1, keepdims=True)
            qdd_task *= np.minimum(1.0, self.max_acc / np.maximum(peak, 1e-12))

            # Largest posture fraction alpha in [0, 1] keeping |qdd_task + alpha * qdd_posture| <= max_acc.
            qdd_posture = qd_posture / self.dt
            bound = np.where(qdd_posture > 0.0, self.max_acc - qdd_task, -self.max_acc - qdd_task)
            ratio = np.where(np.abs(qdd_posture) > 1e-12, bound / np.where(np.abs(qdd_posture) > 1e-12, qdd_posture, 1.0), np.inf)
            alpha = np.clip(np.min(ratio, axis=1, keepdims=True), 0.0, 1.0)
            qdd = qdd_task + alpha * qdd_posture

            u_seqs[:, h] = qdd
            qd = qd + qdd * self.dt
            q = np.clip(q + qd * self.dt, q_min, q_max)

        return u_seqs

    def _create_braking_primitive(self, qd_init: np.ndarray) -> np.ndarray:
        """
        Generates an active braking control sequence: smoothly decelerating joint speeds to 0.
        """
        u_seq = np.zeros((self.H, 7), dtype=np.float32)
        qd = np.asarray(qd_init, dtype=np.float64).copy()

        for h in range(self.H):
            # Proportional braking acceleration
            qdd = -qd / (self.dt * 2.0)
            qdd = np.clip(qdd, -self.max_acc, self.max_acc)
            u_seq[h] = qdd
            qd = qd + qdd * self.dt

        return u_seq
