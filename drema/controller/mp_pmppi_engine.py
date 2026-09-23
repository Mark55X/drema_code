#!/usr/bin/env python
"""
Motion-Primitive Guided Parallel Model Predictive Path Integral (MP-PMPPI) Engine.

Mathematical Implementation of:
- Lelai Zhou et al. (IEEE Transactions on Robotics, Vol. 41, 2025),
  "Parallel MPPI With Gradient-Velocity Modulated SDF Cost for High-Performance Real-Time Dynamic Obstacle Avoidance by Robot Manipulators"
- Mathisen et al. (arXiv 2026), "MP-MPPI: A Motion Primitive Guided Sampling-Based Optimizer for Model Predictive Control"

Core Algorithm Flow:
1. Hybrid Sampling Matrix U_t = [U^{(1), eps}, U^{(2), eps}, ..., U^{(N), eps}, U^{mixed, eps}, U_p] (Eq. 13).
2. Parallel Numerical Rollouts in acceleration space:
   q_{k, h} = q_{k, h-1} + qd_{k, h} * dt, where qd_{k, h} = qd_{k, h-1} + u_{k, h} * dt.
3. Subcost computation:
   - C_goal: Joint distance to IK target (Eq. 26) or Cartesian fallback (Eq. 27).
   - R_s: Sparse Reward attraction bubble (Eq. 12, 28).
   - C_safe: Joint limits and velocity constraints.
   - C_{GVM-SDF}: Gradient-Velocity Modulated collision cost from PyBullet mesh queries (Eq. 10-11).
4. Multi-Planner Local Softmax Weighting: w_{soft}^{(n)} for top-K candidates (Eq. 6, 14, 15).
5. Impartial Judge Evaluation with Log-Sum-Exp Value Function V^{(n)} (Eq. 8, 16).
6. Global Strategy Fusion: Softmax dynamic mixing weights w^{(n)} (Eq. 5, 11).
7. Gaussian Mixture Regression (GMR) policy contraction (Eq. 3, 4, 10).
8. Action extraction and horizon shift.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Top-level physics engine import (PEP 8 compliant, avoided inline imports)
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    p = None
    PYBULLET_AVAILABLE = False

from .franka_kinematics import FrankaKinematics
from .motion_primitives import MotionPrimitiveLibrary


class MPPMPPIEngine:
    """
    MP-PMPPI Real-Time Optimization Engine for Franka Emika Panda.
    """

    def __init__(
        self,
        kinematics: Optional[FrankaKinematics] = None,
        # ---------------------------------------------------------------------
        # Execution & Discretization Parameters:
        # - horizon (H): Lookahead steps.
        #   * On CPU PyBullet: default H = 15 (~0.75s lookahead at dt=0.05s) to guarantee 20-30 Hz rate.
        #   * On GPU PhysX (Isaac Gym): Zhou et al. (Sec VI-2) use H = 30 (~1.5s lookahead).
        #   * When transitioning to GPU PhysX, increase horizon to 30.
        # - dt: Planning timestep delta (seconds) -> 0.05s = 20 Hz control rate.
        # ---------------------------------------------------------------------
        horizon: int = 15,
        dt: float = 0.05,
        # ---------------------------------------------------------------------
        # Sampling & Optimization Parameters:
        # - num_samples_per_planner (M): Number of stochastic trajectories per planner.
        #   * On CPU PyBullet: default M = 24 (total K ~ 80 rollouts) to keep solve latency < 100ms.
        #   * On GPU PhysX (Isaac Gym): Zhou et al. (Sec VI-2) use M = 80 per planner (total K = 580).
        #   * When transitioning to GPU PhysX, increase num_samples_per_planner to 80.
        # - top_k (K_eval): Number of candidate trajectories submitted to the Judge (Eq. 14, 16).
        #   * From Zhou et al. (Sec VI-2): top_k = 20 on GPU. Here capped at min(12, M).
        # - max_joint_acc: Maximum joint acceleration limit [rad/s^2].
        #   * From Zhou et al. (Sec VI-2, Franka Task 3-4): 0.5 rad/s^2 for smooth joint motion.
        # ---------------------------------------------------------------------
        num_samples_per_planner: int = 24,
        top_k: int = 12,
        max_joint_acc: float = 0.5,
        # ---------------------------------------------------------------------
        # Temperature & GMR Parameters (100% from Zhou et al. Section VI-2 & IV):
        # - lambda_param = 1.0 (MPPI exploration temperature)
        # - beta_param = 0.8 (Intramodal softmax temperature, Eq. 6, 15)
        # - alpha_param = 1.0 (Judge strategy differentiation temperature, Eq. 5, 11)
        # - alpha_mu = 0.8, alpha_sigma = 0.2 (GMR learning rates, Eq. 3, 4, 10)
        # ---------------------------------------------------------------------
        lambda_param: float = 1.0,
        beta_param: float = 0.8,
        alpha_param: float = 1.0,
        alpha_mu: float = 0.8,
        alpha_sigma: float = 0.2,
        init_noise_std: float = 0.15,
        # ---------------------------------------------------------------------
        # Task Cost Parameters (100% from Zhou et al. Section V & Table I):
        # - convergence_radius_xi = 0.03m (3 cm attraction bubble for Sparse Reward, Eq. 12, 28)
        # - rho_modulation = 0.8 (GVM-SDF velocity-gradient scaling factor in [0, 1], Eq. 10, 25)
        # ---------------------------------------------------------------------
        convergence_radius_xi: float = 0.03,
        rho_modulation: float = 0.8
    ):
        self.kin = kinematics if kinematics is not None else FrankaKinematics()
        self.H = horizon
        self.dt = dt
        self.M = num_samples_per_planner
        self.top_k = min(top_k, self.M)
        self.max_acc = max_joint_acc

        # Hyperparameters verified against Zhou et al. (2025)
        self.lambda_param = lambda_param
        self.beta = beta_param
        self.alpha = alpha_param
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.init_noise_std = init_noise_std
        self.xi = convergence_radius_xi
        self.rho = rho_modulation

        # SDF potential parameters (Zhou et al. Eq. 21)
        self.sigma_1 = 0.02   # Inscribed safety margin [m]
        self.sigma_2 = 0.15   # Inflation radius [m]
        self.kappa = 15.0     # Descending potential slope

        # Motion Primitives Library (Mathisen et al. 2026)
        self.primitive_lib = MotionPrimitiveLibrary(
            kinematics=self.kin,
            horizon=self.H,
            dt=self.dt,
            max_joint_acc=self.max_acc
        )

        # ---------------------------------------------------------------------
        # Planners Definition & Strategy Weights (Table I in Zhou et al. 2025, Franka)
        # ---------------------------------------------------------------------
        # Weights: [wg (goal), wc (collision), wr (sparse reward), ws (limits/safety)]
        self.strategies = {
            'greedy': {
                'wg': 80.0,
                'wc': 80.0,
                'wr': 200.0,
                'ws': 10.0,
                'use_gvm': False # Greedy uses standard Coll_p (Zhou et al. Table I)
            },
            'sensitive': {
                'wg': 10.0,
                'wc': 250.0,
                'wr': 0.0,      # Sensitive omits goal reward to prioritize avoidance (Table I)
                'ws': 10.0,
                'use_gvm': True  # Sensitive uses Coll_{ppv_theta} (GVM-SDF) (Table I)
            }
        }
        self.planner_names = ['greedy', 'sensitive']
        self.num_planners = len(self.planner_names)

        # The Judge Strategy (pi*): Impartial arbiter (Table I in Zhou et al. 2025)
        self.judge_strategy = {
            'wg': 10.0,
            'wc': 80.0,
            'wr': 80.0,
            'ws': 0.0,
            'use_gvm': False
        }

        # ---------------------------------------------------------------------
        # Distribution Parameters: Means & Covariances over Horizon H [rad/s^2]
        # ---------------------------------------------------------------------
        # Global mixed distribution N(mu_t, Sigma_t)
        self.mu_mixed = np.zeros((self.H, 7), dtype=np.float32)
        self.sigma_mixed = np.ones((self.H, 7), dtype=np.float32) * (self.init_noise_std ** 2)

        # Individual planner distributions N(mu_t^{(n)}, Sigma_t^{(n)})
        self.planner_distributions = {}
        for name in self.planner_names:
            self.planner_distributions[name] = {
                'mu': np.zeros((self.H, 7), dtype=np.float32),
                'sigma': np.ones((self.H, 7), dtype=np.float32) * (self.init_noise_std ** 2)
            }

        # Authority weights assigned by Judge: w^{(n)} (Eq. 5, 11)
        self.mixing_weights = {name: 1.0 / self.num_planners for name in self.planner_names}

        # Cached last IK solution (initialized to None, eliminating redundant boolean flags)
        self.last_ik_solution: Optional[np.ndarray] = None

    def reset(self):
        """Resets the MPC internal policy distributions and cached solutions."""
        self.mu_mixed.fill(0.0)
        self.sigma_mixed.fill(self.init_noise_std ** 2)
        for name in self.planner_names:
            self.planner_distributions[name]['mu'].fill(0.0)
            self.planner_distributions[name]['sigma'].fill(self.init_noise_std ** 2)
        self.mixing_weights = {name: 1.0 / self.num_planners for name in self.planner_names}
        self.last_ik_solution = None

    def solve(
        self,
        q_current: np.ndarray,
        qd_current: np.ndarray,
        target_pos: Optional[np.ndarray],
        target_rot: Optional[np.ndarray] = None,
        digital_twin: Optional[Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes one full iteration of MP-PMPPI.

        :param q_current: Current joint positions [rad] (shape: [7]).
        :param qd_current: Current joint velocities [rad/s] (shape: [7]).
        :param target_pos: Target 3D position [x, y, z] in world frame.
        :param target_rot: Optional target 3x3 orientation matrix.
        :param digital_twin: PyBulletDigitalTwin instance for exact mesh collision checks.
        :return: (optimal_joint_velocity_cmd [shape: 7], diagnostics_dict)
        """
        t0 = time.time()
        q_curr = np.array(q_current, dtype=np.float32)
        qd_curr = np.array(qd_current, dtype=np.float32)

        has_robot = digital_twin is not None and getattr(digital_twin, 'robot_id', -1) >= 0

        # ---------------------------------------------------------------------
        # Step 1: Inverse Kinematics Guidance Target (q_des,t, Eq. 26)
        # ---------------------------------------------------------------------
        q_des_t = None
        if target_pos is not None:
            # Primary IK query: PyBullet native C++ IK
            if has_robot and PYBULLET_AVAILABLE:
                try:
                    ik_target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
                    pb_ik = p.calculateInverseKinematics(
                        digital_twin.robot_id,
                        7, # Franka end-effector link index
                        ik_target,
                        maxNumIterations=20,
                        residualThreshold=1e-3
                    )
                    q_des_t = np.array(pb_ik[:7], dtype=np.float32)
                    self.last_ik_solution = q_des_t.copy()
                except Exception:
                    q_des_t = None

            # Fallback IK query: Analytical DLS IK
            if q_des_t is None:
                q_init_ik = self.last_ik_solution if self.last_ik_solution is not None else q_curr
                q_dls, success = self.kin.solve_dls_ik(q_init_ik, target_pos, target_rot)
                if success:
                    q_des_t = q_dls
                    self.last_ik_solution = q_des_t.copy()

        # ---------------------------------------------------------------------
        # Step 2: Sampling Phase - Construct Hybrid Sampling Matrix U_t (Eq. 13)
        # U_t = [ U^{(1), eps}, U^{(2), eps}, ..., U^{(N), eps}, U^{mixed, eps}, U_p ]
        # ---------------------------------------------------------------------
        sample_batches = []

        # 2a. Intramodal stochastic samples for each planner (Eq. 19)
        for name in self.planner_names:
            mu_n = self.planner_distributions[name]['mu']
            std_n = np.sqrt(np.maximum(self.planner_distributions[name]['sigma'], 1e-6))
            noise = np.random.randn(self.M, self.H, 7).astype(np.float32)
            u_n = mu_n[None, :, :] + noise * std_n[None, :, :]
            sample_batches.append(u_n)

        # 2b. Mixed distribution samples (Eq. 20)
        std_mixed = np.sqrt(np.maximum(self.sigma_mixed, 1e-6))
        noise_mixed = np.random.randn(self.M, self.H, 7).astype(np.float32)
        u_mixed = self.mu_mixed[None, :, :] + noise_mixed * std_mixed[None, :, :]
        sample_batches.append(u_mixed)

        # 2c. Motion Primitives Library U_p (Mathisen et al. 2026)
        obs_list = []
        if has_robot and PYBULLET_AVAILABLE and hasattr(digital_twin, 'tracked_objects'):
            for obj in digital_twin.tracked_objects.values():
                if not obj.get('is_target', False) and obj.get('body_id', -1) >= 0:
                    try:
                        pos, _ = p.getBasePositionAndOrientation(obj['body_id'])
                        obs_list.append({'position': pos})
                    except Exception:
                        pass

        U_p = self.primitive_lib.generate_primitives(
            q_current=q_curr,
            qd_current=qd_curr,
            target_pos=target_pos,
            target_rot=target_rot,
            obstacles=obs_list
        )
        sample_batches.append(U_p)

        # Concatenate into full hybrid sampling matrix U_t: [K, H, 7]
        U_t = np.concatenate(sample_batches, axis=0)
        # Acceleration saturation: clip to [-max_acc, max_acc]
        U_t = np.clip(U_t, -self.max_acc, self.max_acc)
        total_K = U_t.shape[0]

        # ---------------------------------------------------------------------
        # Step 3: Parallel Numerical Rollouts
        # qd_{k, h} = qd_{k, h-1} + u_{k, h} * dt
        # q_{k, h}  = q_{k, h-1}  + qd_{k, h} * dt
        # ---------------------------------------------------------------------
        Q = np.zeros((total_K, self.H, 7), dtype=np.float32)
        QD = np.zeros((total_K, self.H, 7), dtype=np.float32)

        q_prev = np.tile(q_curr, (total_K, 1))
        qd_prev = np.tile(qd_curr, (total_K, 1))

        for h in range(self.H):
            u_h = U_t[:, h, :]
            qd_h = qd_prev + u_h * self.dt
            # Clip joint velocities to physical limits
            qd_h = np.clip(qd_h, -self.kin.QD_MAX, self.kin.QD_MAX)
            q_h = q_prev + qd_h * self.dt
            # Clip joint positions to physical limits
            q_h = np.clip(q_h, self.kin.Q_MIN, self.kin.Q_MAX)

            Q[:, h, :] = q_h
            QD[:, h, :] = qd_h

            q_prev = q_h
            qd_prev = qd_h

        # ---------------------------------------------------------------------
        # Step 4: Subcost Computations (Eq. 17, 26, 27, 28)
        # ---------------------------------------------------------------------
        # 4a. Goal distance cost: dist(x_{i,h})
        goal_costs = np.zeros((total_K, self.H), dtype=np.float32)
        sparse_rewards = np.zeros((total_K, self.H), dtype=np.float32)

        if target_pos is not None:
            if q_des_t is not None:
                # Joint-space goal distance: ||q_{i,h} - q_{des,t}||_2 (Eq. 26)
                diff = Q - q_des_t[None, None, :]
                dist_matrix = np.linalg.norm(diff, axis=-1)
            else:
                # Cartesian goal distance fallback (Eq. 27)
                ee_pos_batch = self.kin.batch_forward_kinematics_ee(Q)
                diff = ee_pos_batch - target_pos[None, None, :]
                dist_matrix = np.linalg.norm(diff, axis=-1)

            goal_costs = dist_matrix.copy()

            # Sparse Reward attraction bubble (Zhou et al. Eq. 12, 28):
            # R_s(x_{i,h}) = 1 - exp(-dist^2 / (2 * xi^2))
            sparse_rewards = 1.0 - np.exp(-(dist_matrix ** 2) / (2.0 * (self.xi ** 2)))

        # 4b. Joint Limit & Dynamics Constraints: C_safe (Eq. 29)
        # Penalizes closeness to joint position and velocity boundaries
        limit_margins_min = np.maximum(0.0, (self.kin.Q_MIN + 0.05)[None, None, :] - Q)
        limit_margins_max = np.maximum(0.0, Q - (self.kin.Q_MAX - 0.05)[None, None, :])
        c_limits = np.sum(limit_margins_min ** 2 + limit_margins_max ** 2, axis=-1)

        # Velocity regularization: penalizes erratic accelerations
        c_acc = np.sum(U_t ** 2, axis=-1)
        safety_costs = c_limits + 0.01 * c_acc

        # 4c. Environmental Collision Costs: Coll_p and GVM-SDF Coll_{ppv_theta} (Eq. 10, 11, 21, 25)
        coll_p_costs, coll_gvm_costs = self._evaluate_collision_costs(
            Q=Q,
            QD=QD,
            digital_twin=digital_twin
        )

        # ---------------------------------------------------------------------
        # Step 5: Multi-Planner Total Cost Calculation (Eq. 18, 29)
        # ---------------------------------------------------------------------
        total_costs = {}
        for name in self.planner_names:
            st = self.strategies[name]
            coll_term = coll_gvm_costs if st['use_gvm'] else coll_p_costs
            # Total trajectory cost: sum over horizon H
            step_cost = (
                st['wg'] * goal_costs +
                st['wc'] * coll_term +
                st['wr'] * sparse_rewards +
                st['ws'] * safety_costs
            )
            total_costs[name] = np.sum(step_cost, axis=1) # Shape: [total_K]

        # Judge impartial cost Cost_{pi*} (Eq. 13)
        st_judge = self.judge_strategy
        coll_judge = coll_p_costs
        step_cost_judge = (
            st_judge['wg'] * goal_costs +
            st_judge['wc'] * coll_judge +
            st_judge['wr'] * sparse_rewards +
            st_judge['ws'] * safety_costs
        )
        judge_costs = np.sum(step_cost_judge, axis=1) # Shape: [total_K]

        # ---------------------------------------------------------------------
        # Step 6: Selective Primitive Filtering & Top-K Softmax (Eq. 6, 14, 15)
        # w_{soft}^{(n)}(x_k) = softmax_{beta}(-Cost^{(n)}(x_k))
        # ---------------------------------------------------------------------
        topk_indices = {}
        w_soft = {}
        V_n = {} # Judge Value Function for planner n

        for name in self.planner_names:
            c_n = total_costs[name]
            # Select top-K lowest cost candidates for this planner
            idx_sorted = np.argsort(c_n)[:self.top_k]
            topk_indices[name] = idx_sorted

            # Intramodal softmax weights with temperature beta (Eq. 6, 15)
            c_topk = c_n[idx_sorted]
            c_min = np.min(c_topk)
            exp_weights = np.exp(-(c_topk - c_min) / max(self.beta, 1e-4))
            weights = exp_weights / np.maximum(np.sum(exp_weights), 1e-8)
            w_soft[name] = weights

            # -----------------------------------------------------------------
            # Step 7: Objective Judge Value Function via Log-Sum-Exp (Eq. 8, 16)
            # V^{(n)} = -lambda * log sum exp(A_k - A_max) - lambda * A_max
            #           + lambda * log sum exp(B_k - B_max) + lambda * B_max
            # where A_k = -1/beta * Cost^{(n)} - 1/lambda * Cost_{pi*}
            #       B_k = -1/beta * Cost^{(n)}
            # -----------------------------------------------------------------
            judge_topk = judge_costs[idx_sorted]
            A_k = -(1.0 / self.beta) * c_topk - (1.0 / self.lambda_param) * judge_topk
            B_k = -(1.0 / self.beta) * c_topk

            A_max = np.max(A_k)
            B_max = np.max(B_k)

            sum_exp_A = np.sum(np.exp(A_k - A_max))
            sum_exp_B = np.sum(np.exp(B_k - B_max))

            v_val = (
                -self.lambda_param * np.log(np.maximum(sum_exp_A, 1e-8)) - self.lambda_param * A_max
                + self.lambda_param * np.log(np.maximum(sum_exp_B, 1e-8)) + self.lambda_param * B_max
            )
            V_n[name] = float(v_val)

            # Update planner's own distribution using weighted top-K controls (Eq. 3, 4)
            u_topk = U_t[idx_sorted] # [top_k, H, 7]
            weighted_mu = np.sum(weights[:, None, None] * u_topk, axis=0) # [H, 7]
            diff_u = u_topk - weighted_mu[None, :, :]
            weighted_sigma = np.sum(weights[:, None, None] * (diff_u ** 2), axis=0) # [H, 7]

            self.planner_distributions[name]['mu'] = (
                (1.0 - self.alpha_mu) * self.planner_distributions[name]['mu'] + self.alpha_mu * weighted_mu
            )
            self.planner_distributions[name]['sigma'] = (
                (1.0 - self.alpha_sigma) * self.planner_distributions[name]['sigma'] + self.alpha_sigma * weighted_sigma
            )

        # ---------------------------------------------------------------------
        # Step 8: Global Strategy Dynamic Mixing Weights w^{(n)} (Eq. 5, 11)
        # w^{(n)} = softmax_{alpha}(-V^{(n)})
        # ---------------------------------------------------------------------
        v_array = np.array([V_n[name] for name in self.planner_names], dtype=np.float32)
        v_min = np.min(v_array)
        exp_v = np.exp(-(v_array - v_min) / max(self.alpha, 1e-4))
        global_w = exp_v / np.maximum(np.sum(exp_v), 1e-8)

        for i, name in enumerate(self.planner_names):
            self.mixing_weights[name] = float(global_w[i])

        # ---------------------------------------------------------------------
        # Step 9: Gaussian Mixture Regression (GMR) Fusion (Eq. 3, 4, 10)
        # mu_{t,h} = (1 - alpha_mu) * mu_{t-1,h} + alpha_mu * sum w^{(n)} * mu_{t,h}^{(n)}
        # ---------------------------------------------------------------------
        gmr_mu_sum = np.zeros((self.H, 7), dtype=np.float32)
        gmr_sigma_sum = np.zeros((self.H, 7), dtype=np.float32)

        for name in self.planner_names:
            w_i = self.mixing_weights[name]
            mu_i = self.planner_distributions[name]['mu']
            sigma_i = self.planner_distributions[name]['sigma']
            gmr_mu_sum += w_i * mu_i
            gmr_sigma_sum += w_i * (sigma_i + mu_i ** 2)

        self.mu_mixed = (1.0 - self.alpha_mu) * self.mu_mixed + self.alpha_mu * gmr_mu_sum
        self.sigma_mixed = (
            (1.0 - self.alpha_sigma) * self.sigma_mixed
            - self.alpha_sigma * (self.mu_mixed ** 2)
            + self.alpha_sigma * gmr_sigma_sum
        )
        self.sigma_mixed = np.maximum(self.sigma_mixed, 1e-6)

        # ---------------------------------------------------------------------
        # Step 10: Action Selection and Horizon Shift (Algorithm from paper, Lines 28-31)
        # ---------------------------------------------------------------------
        # Optimal acceleration command is first step of blended sequence: u_t^* = mu_{t, 1}
        u_optimal_acc = self.mu_mixed[0].copy()

        # Integrated velocity command: qd_{cmd} = qd_{curr} + u_t^* * dt
        qd_optimal = qd_curr + u_optimal_acc * self.dt
        # Project onto velocity limits (Algorithm from paper, Line 11)
        qd_optimal = np.clip(qd_optimal, -self.kin.QD_MAX, self.kin.QD_MAX)

        # Shift distributions forward across horizon (Warm Starting)
        self.mu_mixed[:-1] = self.mu_mixed[1:]
        self.mu_mixed[-1] = self.mu_mixed[-2]
        self.sigma_mixed[:-1] = self.sigma_mixed[1:]
        self.sigma_mixed[-1] = self.sigma_mixed[-2]

        for name in self.planner_names:
            self.planner_distributions[name]['mu'][:-1] = self.planner_distributions[name]['mu'][1:]
            self.planner_distributions[name]['mu'][-1] = self.planner_distributions[name]['mu'][-2]
            self.planner_distributions[name]['sigma'][:-1] = self.planner_distributions[name]['sigma'][1:]
            self.planner_distributions[name]['sigma'][-1] = self.planner_distributions[name]['sigma'][-2]

        calc_time_ms = (time.time() - t0) * 1000.0

        diagnostics = {
            'calc_time_ms': calc_time_ms,
            'weights': self.mixing_weights.copy(),
            'judge_values': V_n.copy(),
            'best_greedy_idx': int(topk_indices['greedy'][0]),
            'best_sensitive_idx': int(topk_indices['sensitive'][0]),
            'total_candidates': total_K,
            'q_des_available': (q_des_t is not None)
        }

        return qd_optimal, diagnostics

    def _evaluate_collision_costs(
        self,
        Q: np.ndarray,
        QD: np.ndarray,
        digital_twin: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates exact mesh-level collision distance and GVM-SDF modulation.

        Mathematical Foundations:
        -------------------------
        1. SDF Workspace Potential Function Phi(d) - Zhou et al. (IEEE T-RO 2025), Eq. (21):
           Phi(d) = 1.0                              if d < sigma_1
                    exp(-kappa * (d - sigma_1))      if sigma_1 <= d <= sigma_2
                    0.0                              otherwise
           where:
             - d: Minimum signed distance between the robot link and obstacle surface [m].
             - sigma_1: Inscribed safety radius / margin (self.sigma_1 = 0.02m).
             - sigma_2: Inflation boundary radius (self.sigma_2 = 0.15m).
             - kappa: Descending slope constant of the safety margin (self.kappa = 15.0).

        2. Static Distance Potential Cost Coll_p - Zhou et al., Eq. (25):
           Coll_p(x_{i,h}) = sum_{k=1}^L Phi(pt_k^*)
           Used by the Greedy strategy (pi_greedy) and Judge (pi*).

        3. Gradient-Velocity Modulated SDF Cost Coll_{ppv_theta} - Zhou et al., Eq. (10) & (25):
           Coll_{ppv_theta}(x_{i,h}) = sum_{k=1}^L [ Phi(pt_k^*) + Phi(pt_k^*) * ||Vel(pt_k^*)|| * (1 - rho * cos(theta)) ]
           where:
             - Vel(pt_k^*): Cartesian velocity of the robot link control point, computed via
               geometric Jacobian: Vel = J_v(q) * q_dot (Zhou et al., Eq. 24).
             - rho: Modulation scaling factor in [0, 1] (self.rho = 0.8).
             - theta: Angle between link velocity Vel and the obstacle escape gradient nabla Phi:
               cos(theta) = (nabla Phi * Vel) / (||nabla Phi|| * ||Vel||) (Zhou et al., Eq. 11).
               * If moving away from obstacle (theta < pi/2 => cos(theta) > 0): cost decreases!
               * If moving towards obstacle (theta -> pi => cos(theta) < 0): cost is amplified!

        Analytical Distance Gradient without Voxel Grid:
        ------------------------------------------------
        In Zhou et al., nabla Phi was computed via finite differencing on a 3D GPU voxel grid.
        Here, we leverage PyBullet's exact GJK/EPA contact query: pt[7] ('contactNormalOnB') is
        the unit normal on the obstacle surface pointing directly towards the robot.
        This normal is mathematically identical to the normalized escape gradient:
            nabla Phi / ||nabla Phi|| == contactNormalOnB
        This provides an exact, continuous spatial gradient directly from 3D meshes without
        voxelization artifacts or memory overhead.

        IMPORTANT NOTE ON CPU OPTIMIZATION & GPU MIGRATION:
        ---------------------------------------------------
        - CPU Mode (Current PyBullet C++):
          Evaluating all K * H candidate states (e.g. 80 * 15 = 1200 physics queries) sequentially
          in Python takes ~250 ms. To maintain a real-time 20 Hz control rate (<100 ms), we evaluate
          PyBullet collision checks at key lookahead waypoints (step_stride = max(1, H // 5), e.g.
          5 waypoints across the horizon) and interpolate across intermediate steps.
        - GPU Migration (Future Isaac Gym / PhysX GPU / CuRobo):
          When moving simulation to GPU where thousands of rollouts execute concurrently on CUDA cores,
          set `step_stride = 1` to query collisions on all lookahead timesteps simultaneously.

        :param Q: Candidate joint positions of shape [K, H, 7].
        :param QD: Candidate joint velocities of shape [K, H, 7].
        :param digital_twin: PyBulletDigitalTwin instance with tracked meshes.
        :return: (coll_p_costs [K, H], coll_gvm_costs [K, H])
        """
        K, H, _ = Q.shape
        coll_p = np.zeros((K, H), dtype=np.float32)
        coll_gvm = np.zeros((K, H), dtype=np.float32)

        has_robot = digital_twin is not None and getattr(digital_twin, 'robot_id', -1) >= 0
        if not has_robot or not PYBULLET_AVAILABLE or len(digital_twin.tracked_objects) == 0:
            return coll_p, coll_gvm

        # Extract dynamic obstacle body IDs (ignore target manipulation objects)
        obstacle_body_ids = [
            obj['body_id'] for obj in digital_twin.tracked_objects.values()
            if not obj.get('is_target', False) and obj.get('body_id', -1) >= 0
        ]
        if not obstacle_body_ids:
            return coll_p, coll_gvm

        robot_id = digital_twin.robot_id

        # CPU OPTIMIZATION: Subsample lookahead waypoints to sustain 20 Hz control rate.
        # On GPU PhysX, change step_stride to 1 to evaluate every lookahead step.
        step_stride = max(1, H // 5)
        eval_steps = list(range(0, H, step_stride))
        if (H - 1) not in eval_steps:
            eval_steps.append(H - 1)

        for h in eval_steps:
            for k in range(K):
                q_step = Q[k, h]
                qd_step = QD[k, h]

                # Update Franka Panda joint positions in PyBullet for this candidate state
                num_j = p.getNumJoints(robot_id)
                arm_j = 0
                for j_idx in range(num_j):
                    info = p.getJointInfo(robot_id, j_idx)
                    # Joints 0..6 in panda.urdf correspond to the 7 arm revolute joints
                    if info[2] != p.JOINT_FIXED and arm_j < 7:
                        p.resetJointState(robot_id, j_idx, float(q_step[arm_j]))
                        arm_j += 1

                # Update broadphase/narrowphase collision pairs (GJK/EPA between exact meshes)
                p.performCollisionDetection()

                step_coll_p = 0.0
                step_coll_gvm = 0.0

                for obs_id in obstacle_body_ids:
                    # Query closest points within the inflation boundary (distance <= sigma_2)
                    closest_pts = p.getClosestPoints(
                        bodyA=robot_id,
                        bodyB=obs_id,
                        distance=float(self.sigma_2)
                    )
                    if not closest_pts:
                        continue

                    for pt in closest_pts:
                        # PyBullet API Contact Point Tuple Indices:
                        # pt[3]: linkIndexA (Index of the robot link closest to the obstacle)
                        # pt[7]: contactNormalOnB (Vector on obstacle surface pointing towards robot link)
                        # pt[8]: contactDistance (Signed Euclidean clearance distance in meters; <0 is penetration)
                        d = pt[8]

                        # 1. SDF Potential Phi(d) (Zhou et al., Eq. 21)
                        if d < self.sigma_1:
                            phi = 1.0
                        elif d <= self.sigma_2:
                            phi = float(np.exp(-self.kappa * (d - self.sigma_1)))
                        else:
                            phi = 0.0

                        # Numerical cutoff for negligible potential
                        if phi <= 1e-4:
                            continue

                        step_coll_p += phi

                        # 2. Distance Gradient Vector nabla Phi (from PyBullet contactNormalOnB)
                        normal_grad = np.array(pt[7], dtype=np.float32) # [nx, ny, nz]
                        norm_mag = np.linalg.norm(normal_grad)
                        if norm_mag > 1e-6: # Numerical epsilon to prevent division by zero
                            normal_grad = normal_grad / norm_mag

                        # 3. Cartesian Velocity of the Closest Link: Vel(pt_k^*) = J_v * qd (Zhou et al., Eq. 24)
                        link_a = pt[3]
                        # Map PyBullet Franka link index: 0..6 -> arm links 1..7; >=7 -> flange/hand/fingers
                        fk_link_idx = -1 if (link_a < 0 or link_a >= 7) else (link_a + 1)
                        vel_cart = self.kin.compute_cartesian_velocity(q_step, qd_step, link_idx=fk_link_idx)
                        vel_mag = np.linalg.norm(vel_cart)

                        # 4. Cosine Alignment theta between velocity and escape gradient (Zhou et al., Eq. 11)
                        if vel_mag > 1e-4: # Numerical epsilon for stationary robot link
                            cos_theta = float(np.dot(normal_grad, vel_cart) / (norm_mag * vel_mag))
                        else:
                            cos_theta = 0.0

                        # 5. GVM-SDF Modulation Term (Zhou et al., Eq. 10 & 25)
                        # (1 - rho * cos(theta)): decreases when escaping (cos>0), amplifies when approaching (cos<0)
                        modulation = 1.0 - self.rho * cos_theta
                        gvm_term = phi * (1.0 + vel_mag * modulation)
                        step_coll_gvm += gvm_term

                coll_p[k, h] = step_coll_p
                coll_gvm[k, h] = step_coll_gvm

        # Interpolate across un-evaluated lookahead steps along horizon
        for h in range(H):
            if h not in eval_steps:
                prev_h = max([s for s in eval_steps if s <= h], default=0)
                coll_p[:, h] = coll_p[:, prev_h]
                coll_gvm[:, h] = coll_gvm[:, prev_h]

        return coll_p, coll_gvm
