#!/usr/bin/env python
"""
PyBullet Digital Twin for DREMA Dynamic Inference System.

Agnostic & Dynamic Scene Manager:
- Only the static workcell baseline is loaded at startup (Ground plane, Table workspace, Franka Panda robot).
- All scene objects (obstacles, containers, manipulated objects, target) are dynamically spawned
  and updated at runtime as they are discovered and reconstructed by the Perception pipeline.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any


import pybullet as p
import pybullet_data
from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR


class PyBulletDigitalTwin(BaseDigitalTwin):
    """
    Physical Digital Twin of the manipulation scene running in PyBullet.
    """

    def __init__(
        self,
        visualize: bool = False,
        table_z: float = 0.75,
        robot_urdf_path: Optional[str] = None,
        tracking_mode: str = "constraint",
        constraint_max_force: float = 300.0,
        kp_pos: float = 250.0,
        kd_pos: float = 30.0,
        kp_rot: float = 15.0,
        kd_rot: float = 1.5,
        sim_substeps: int = 1
    ):
        self.visualize = visualize
        self.table_z = table_z
        self.tracking_mode = tracking_mode.lower()
        self.constraint_max_force = float(constraint_max_force)
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot)
        self.kd_rot = float(kd_rot)
        self.sim_substeps = max(1, int(sim_substeps))

        self.client_id = -1
        self.robot_id = -1
        self.table_id = -1

        # Dynamic registry of objects spawned by perception (obj_id -> object metadata)
        self.tracked_objects: Dict[int, Dict] = {}


        # Connect to PyBullet
        if self.visualize:
            self.client_id = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.25, 0.0, self.table_z]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)

        # 1. Base Workcell Environment: Ground Plane (Table is spawned dynamically from t=0 scanning)
        p.loadURDF("plane.urdf")

        # 2. Cache Robot URDF path (Franka Panda is loaded dynamically upon first communication!)
        if robot_urdf_path is None:
            robot_urdf_path = "franka_panda/panda.urdf"
        self.robot_urdf_path = robot_urdf_path

    def load_robot(
        self,
        base_position: Tuple[float, float, float],
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        joint_positions: Optional[List[float]] = None
    ) -> bool:
        """
        Dynamically loads or updates Franka Panda robot in PyBullet at the given world position.
        Returns True if loaded/updated successfully, False otherwise.
        """
        if self.client_id < 0:
            return False

        if self.robot_id >= 0:
            try:
                p.resetBasePositionAndOrientation(self.robot_id, list(base_position), list(base_orientation))
                if joint_positions is not None:
                    self.sync_robot_state(joint_positions)
                return True
            except Exception as e:
                print(f"[PYBULLET DIGITAL TWIN WARNING] Failed to update robot pose: {e}")
                return False

        # Load URDF at the exact base position received dynamically
        try:
            self.robot_id = p.loadURDF(self.robot_urdf_path, list(base_position), list(base_orientation), useFixedBase=True)
            print(f"[PYBULLET DIGITAL TWIN] Dynamically loaded Franka Panda URDF at [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}] (ID: {self.robot_id})")
        except Exception as e:
            print(f"[PYBULLET DIGITAL TWIN ERROR] Failed to load Franka Panda URDF: {e}")
            self.robot_id = -1
            return False

        if joint_positions is not None:
            self.sync_robot_state(joint_positions)

        return True

    def set_robot_base_pose(
        self,
        base_position: Tuple[float, float, float],
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)
    ):
        """Updates or dynamically loads the robot base position and orientation in PyBullet."""
        if self.robot_id < 0:
            self.load_robot(base_position, base_orientation)
        else:
            try:
                p.resetBasePositionAndOrientation(self.robot_id, list(base_position), list(base_orientation))
            except Exception as e:
                print(f"[PYBULLET DIGITAL TWIN WARNING] Failed to reset robot base pose: {e}")

    def spawn_scanned_table(
        self,
        table_z: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        mesh_file_path: Optional[str] = None,
        color: Tuple[float, float, float, float] = DEFAULT_TABLE_COLOR
    ) -> int:
        """
        Dynamically spawns the tabletop workspace surface from real 3D scanning data at t=0.
        Accepts either an extracted 3D mesh or planar bounding box (xmin, xmax, ymin, ymax).
        Does NOT rely on hardcoded table geometry!
        """
        if self.client_id < 0:
            return -1

        # Remove existing table if re-scanning upon episode reset
        if self.table_id >= 0:
            try:
                p.removeBody(self.table_id)
            except Exception:
                pass
            self.table_id = -1

        self.table_z = table_z

        if mesh_file_path and os.path.exists(mesh_file_path):
            try:
                col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file_path)
                vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file_path, rgbaColor=list(color))
                self.table_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[0, 0, 0]
                )
                print(f"[PYBULLET DIGITAL TWIN] Spawned scanned table from 3D surface mesh: {mesh_file_path} (ID: {self.table_id})")
                return self.table_id
            except Exception as e:
                print(f"[PYBULLET DIGITAL TWIN WARNING] Failed to load table mesh {mesh_file_path}: {e}. Falling back to planar bounds if available.")

        # Spawn table structure extending from ground Z=0 to surface Z=table_z
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            hx = max(0.01, (xmax - xmin) / 2.0)
            hy = max(0.01, (ymax - ymin) / 2.0)

            # Table body reaches from Z=0 to table_z
            hz = self.table_z / 2.0
            table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(color))
            self.table_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_col,
                baseVisualShapeIndex=table_vis,
                basePosition=[cx, cy, hz]
            )
            print(f"[PYBULLET DIGITAL TWIN] Spawned scanned table structure with bounds X[{cx-hx:.2f}, {cx+hx:.2f}], Y[{cy-hy:.2f}, {cy+hy:.2f}], Z=[0.0, {self.table_z:.3f}]m (ID: {self.table_id})")
            return self.table_id

        print("[PYBULLET DIGITAL TWIN WARNING] Neither valid mesh_file_path nor table bounds provided. Table not spawned.")
        return -1


    # -------------------------------------------------------------------------
    # Generic Dynamic Object Spawning (Called by Perception pipeline)
    # -------------------------------------------------------------------------

    def spawn_scanned_mesh_obstacle(
        self,
        mesh_path: str,
        initial_pos: Tuple[float, float, float],
        initial_quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "obstacle",
        mass: float = 1.0,
        color: Optional[Union[List[float], Tuple[float, ...]]] = None,
        obj_id: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Dynamically spawns a real 3D surface mesh obstacle extracted by Perception at t=0
        into the PyBullet Digital Twin with realistic rigid body physical dynamics.
        """
        if self.client_id < 0:
            return -1

        target_obj_id = obj_id if obj_id is not None else len(self.tracked_objects)

        if target_obj_id in self.tracked_objects:
            self.remove_object(target_obj_id)

        col = tuple(color) if color is not None else DEFAULT_OBSTACLE_COLOR

        try:
            if mass <= 0.0 and self.tracking_mode == "constraint":
                print(f"[PYBULLET DIGITAL TWIN ERROR] Obstacle '{name}' (ID: {target_obj_id}) spawned with mass={mass} <= 0! In PyBullet, static bodies cannot be moved by constraints.")

            col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path)
            vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, rgbaColor=list(col))
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=list(initial_pos),
                baseOrientation=list(initial_quat)
            )

            # Configure realistic physical contact dynamics (friction, non-bouncy restitution)
            if mass > 0:
                p.changeDynamics(
                    body_id,
                    -1,
                    lateralFriction=0.5,
                    spinningFriction=0.01,
                    rollingFriction=0.001,
                    restitution=0.1
                )

            self.tracked_objects[target_obj_id] = {
                'body_id': body_id,
                'mesh_path': mesh_path,
                'name': name,
                'mass': mass,
                'color': col,
                'canonical_position': initial_pos,
                'target_pos': initial_pos,
                'target_quat': initial_quat
            }

            # Initialize 6-DoF tracking constraint immediately if in constraint mode
            if self.tracking_mode == "constraint":
                cid = p.createConstraint(
                    parentBodyUniqueId=body_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0.0, 0.0, 0.0],
                    parentFramePosition=[0.0, 0.0, 0.0],
                    childFramePosition=list(initial_pos),
                    childFrameOrientation=list(initial_quat)
                )
                p.changeConstraint(cid, maxForce=self.constraint_max_force)
                self.tracked_objects[target_obj_id]['constraint_id'] = cid

            print(f"[PYBULLET DIGITAL TWIN] Spawned mesh obstacle ID {target_obj_id} ('{name}', Mass={mass}kg, PyBullet Body ID: {body_id})")
            return body_id
        except Exception as e:
            print(f"[PYBULLET DIGITAL TWIN ERROR] Failed to spawn mesh obstacle {target_obj_id}: {e}")
            return -1

    def spawn_mesh_object(self, *args, **kwargs) -> int:
        """Backward-compatible alias for spawn_scanned_mesh_obstacle."""
        if 'mesh_file_path' in kwargs:
            kwargs['mesh_path'] = kwargs.pop('mesh_file_path')
        if 'initial_position' in kwargs:
            kwargs['initial_pos'] = kwargs.pop('initial_position')
        if 'initial_orientation' in kwargs:
            kwargs['initial_quat'] = kwargs.pop('initial_orientation')
        return self.spawn_scanned_mesh_obstacle(*args, **kwargs)

    def remove_object(self, obj_id: int):
        """Removes an object from PyBullet and cleans up its tracking constraint if present."""
        if self.client_id < 0:
            return
        if obj_id in self.tracked_objects:
            body_id = self.tracked_objects[obj_id]['body_id']
            constraint_id = self.tracked_objects[obj_id].get('constraint_id', -1)
            if constraint_id >= 0:
                try:
                    p.removeConstraint(constraint_id)
                except Exception:
                    pass
            p.removeBody(body_id)
            del self.tracked_objects[obj_id]

    def clear_dynamic_objects(self):
        """Removes all dynamically discovered objects upon reset."""
        for obj_id in list(self.tracked_objects.keys()):
            self.remove_object(obj_id)
        self.tracked_objects.clear()

    # -------------------------------------------------------------------------
    # State Synchronization & Physics Queries
    # -------------------------------------------------------------------------

    def sync_robot_state(self, joint_positions: List[float]):
        """Synchronizes robot joint angles in the Digital Twin from CoppeliaSim."""
        if self.robot_id < 0:
            return

        num_joints = p.getNumJoints(self.robot_id)
        arm_j = 0
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED and arm_j < len(joint_positions):
                p.resetJointState(self.robot_id, i, joint_positions[arm_j])
                arm_j += 1

    def sync_object_pose(
        self,
        obj_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float]
    ):
        """
        Synchronizes the 3D position and orientation of an object tracked by RecurGS SE(3).
        Operates dynamically according to the selected tracking_mode:
        - "constraint": 6-DoF constraint attaching rigid body to world target with finite maxForce.
        - "pd_force": Virtual spring-damper PD force/torque with feedforward gravity compensation.
        - "teleport": Hard kinematic teleportation via resetBasePositionAndOrientation.
        """
        if obj_id not in self.tracked_objects or self.client_id < 0:
            return

        obj_data = self.tracked_objects[obj_id]
        body_id = obj_data['body_id']
        pos = (float(position[0]), float(position[1]), float(position[2]))
        orn = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))

        obj_data['target_pos'] = pos
        obj_data['target_quat'] = orn

        if self.tracking_mode == "constraint":
            mass = float(obj_data.get('mass', 1.0))
            if mass <= 0.0:
                print(f"[PYBULLET DIGITAL TWIN ERROR] Cannot sync pose for obstacle ID {obj_id} with mass={mass} <= 0 using constraints! Body will NOT move in PyBullet.")
                return
            cid = obj_data.get('constraint_id', -1)
            if cid >= 0:
                p.changeConstraint(
                    cid,
                    jointChildPivot=list(pos),
                    jointChildFrameOrientation=list(orn),
                    maxForce=self.constraint_max_force
                )
        elif self.tracking_mode == "pd_force":
            # Target stored, forces applied dynamically in step()
            pass
        elif self.tracking_mode == "teleport":
            p.resetBasePositionAndOrientation(body_id, list(pos), list(orn))

    def get_object_pose(
        self,
        obj_id: int
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Retrieves rigid body position and orientation for the given object ID."""
        if obj_id in self.tracked_objects and self.client_id >= 0:
            body_id = self.tracked_objects[obj_id]['body_id']
            pos, orn = p.getBasePositionAndOrientation(body_id)
            return tuple(pos), tuple(orn)
        return None

    def get_min_obstacle_distance(self) -> float:
        """
        Calculates the minimum clearance distance between the Franka Panda arm and
        ANY currently tracked dynamic obstacle in the scene.
        Used by the MPC controller for proactive collision avoidance.
        """
        if self.robot_id < 0 or len(self.tracked_objects) == 0:
            return float('inf')

        min_distance = float('inf')

        for obj_id, obj_data in self.tracked_objects.items():
            body_id = obj_data['body_id']
            contact_pts = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=body_id,
                distance=2.0
            )

            if contact_pts:
                obj_min_d = min(pt[8] for pt in contact_pts)
                if obj_min_d < min_distance:
                    min_distance = obj_min_d

        return float(min_distance)

    def step(self):
        """Advances physical simulation forward (with sub-stepping and PD tracking if active)."""
        if self.client_id < 0:
            return

        for _ in range(self.sim_substeps):
            if self.tracking_mode == "pd_force":
                self._apply_pd_tracking_forces()
            p.stepSimulation()

    def _apply_pd_tracking_forces(self):
        """Applies virtual spring-damper PD forces and torques for 'pd_force' tracking mode."""
        for obj_id, obj_data in self.tracked_objects.items():
            if 'target_pos' not in obj_data:
                continue

            body_id = obj_data['body_id']
            mass = float(obj_data.get('mass', 1.0))
            target_pos = np.array(obj_data['target_pos'], dtype=np.float32)
            target_quat = np.array(obj_data['target_quat'], dtype=np.float32)

            curr_pos_t, curr_quat_t = p.getBasePositionAndOrientation(body_id)
            lin_vel_t, ang_vel_t = p.getBaseVelocity(body_id)

            curr_pos = np.array(curr_pos_t, dtype=np.float32)
            curr_quat = np.array(curr_quat_t, dtype=np.float32)
            lin_vel = np.array(lin_vel_t, dtype=np.float32)
            ang_vel = np.array(ang_vel_t, dtype=np.float32)

            # 1. Linear PD force with feedforward gravity compensation (m * g)
            grav_comp = np.array([0.0, 0.0, mass * 9.81], dtype=np.float32)
            force = self.kp_pos * (target_pos - curr_pos) - self.kd_pos * lin_vel + grav_comp
            p.applyExternalForce(body_id, -1, force.tolist(), curr_pos.tolist(), p.WORLD_FRAME)

            # 2. Rotational PD torque via quaternion difference: q_rel = q_target * inv(q_curr)
            q_inv = np.array([-curr_quat[0], -curr_quat[1], -curr_quat[2], curr_quat[3]], dtype=np.float32)
            w1, x1, y1, z1 = target_quat[3], target_quat[0], target_quat[1], target_quat[2]
            w2, x2, y2, z2 = q_inv[3], q_inv[0], q_inv[1], q_inv[2]
            qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
            qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
            qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
            qz = w1*z2 + x1*y2 - y1*x2 + z1*w2

            q_rel_vec = np.array([qx, qy, qz], dtype=np.float32)
            if qw < 0.0:
                q_rel_vec = -q_rel_vec  # Shortest geodesic path on S^3

            rot_error = 2.0 * q_rel_vec
            torque = self.kp_rot * rot_error - self.kd_rot * ang_vel
            p.applyExternalTorque(body_id, -1, torque.tolist(), p.WORLD_FRAME)

    def get_tracked_obstacles_info(self) -> List[Dict[str, Any]]:
        """Retrieves list of tracked dynamic obstacle dictionaries."""
        obstacles = []
        if self.client_id < 0 or len(self.tracked_objects) == 0:
            return obstacles

        for obj_id, obj_data in self.tracked_objects.items():
            body_id = obj_data.get('body_id', -1)
            pos = None
            orn = None
            if body_id >= 0:
                try:
                    p_pos, p_orn = p.getBasePositionAndOrientation(body_id)
                    pos = tuple(float(x) for x in p_pos)
                    orn = tuple(float(x) for x in p_orn)
                except Exception:
                    pass
            if pos is None and 'target_pos' in obj_data:
                pos = tuple(float(x) for x in obj_data['target_pos'])
                orn = tuple(float(x) for x in obj_data.get('target_quat', (0, 0, 0, 1)))

            if pos is not None:
                obstacles.append({
                    'id': obj_id,
                    'body_id': body_id,
                    'name': obj_data.get('name', f"obstacle_{obj_id}"),
                    'position': pos,
                    'orientation': orn if orn is not None else (0.0, 0.0, 0.0, 1.0),
                    'is_target': bool(obj_data.get('is_target', False))
                })
        return obstacles

    def calculate_inverse_kinematics(
        self,
        target_pos: Tuple[float, float, float],
        target_quat: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """Calculates inverse kinematics solution using PyBullet C++ solver."""
        if self.client_id < 0 or self.robot_id < 0:
            return None
        try:
            ik_target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
            kwargs = {
                "maxNumIterations": 20,
                "residualThreshold": 1e-3
            }
            if target_quat is not None:
                kwargs["targetOrientation"] = [float(x) for x in target_quat]

            pb_ik = p.calculateInverseKinematics(
                self.robot_id,
                7,  # Franka end-effector link index
                ik_target,
                **kwargs
            )
            return np.array(pb_ik[:7], dtype=np.float32)
        except Exception:
            return None

    def compute_trajectory_collision_costs(
        self,
        Q: np.ndarray,
        QD: np.ndarray,
        sigma_1: float,
        sigma_2: float,
        kappa: float,
        rho: float,
        kin_helper: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates collision clearance and GVM (Gradient-Velocity Modulated) costs for
        a batch of candidate trajectories Q (K x H x 7) and QD (K x H x 7) using PyBullet GJK/EPA.
        """
        K, H, _ = Q.shape
        coll_p = np.zeros((K, H), dtype=np.float32)
        coll_gvm = np.zeros((K, H), dtype=np.float32)

        if self.client_id < 0 or self.robot_id < 0 or len(self.tracked_objects) == 0:
            return coll_p, coll_gvm

        obstacle_body_ids = [
            obj['body_id'] for obj in self.tracked_objects.values()
            if not obj.get('is_target', False) and obj.get('body_id', -1) >= 0
        ]
        if not obstacle_body_ids:
            return coll_p, coll_gvm

        robot_id = self.robot_id

        # Subsample lookahead waypoints to sustain high control frequency
        step_stride = max(1, H // 5)
        eval_steps = list(range(0, H, step_stride))
        if (H - 1) not in eval_steps:
            eval_steps.append(H - 1)

        num_j = p.getNumJoints(robot_id)
        # Pre-cache non-fixed arm joint indices
        arm_joint_indices = []
        for j_idx in range(num_j):
            info = p.getJointInfo(robot_id, j_idx)
            if info[2] != p.JOINT_FIXED and len(arm_joint_indices) < 7:
                arm_joint_indices.append(j_idx)

        for h in eval_steps:
            for k in range(K):
                q_step = Q[k, h]
                qd_step = QD[k, h]

                # Update Franka Panda joint positions in PyBullet
                for arm_j, j_idx in enumerate(arm_joint_indices):
                    p.resetJointState(robot_id, j_idx, float(q_step[arm_j]))

                p.performCollisionDetection()

                step_coll_p = 0.0
                step_coll_gvm = 0.0

                for obs_id in obstacle_body_ids:
                    closest_pts = p.getClosestPoints(
                        bodyA=robot_id,
                        bodyB=obs_id,
                        distance=float(sigma_2)
                    )
                    if not closest_pts:
                        continue

                    for pt in closest_pts:
                        d = pt[8]

                        # 1. SDF Potential Phi(d)
                        if d < sigma_1:
                            phi = 1.0
                        elif d <= sigma_2:
                            phi = float(np.exp(-kappa * (d - sigma_1)))
                        else:
                            phi = 0.0

                        if phi <= 1e-4:
                            continue

                        step_coll_p += phi

                        # 2. Distance Gradient Vector nabla Phi
                        normal_grad = np.array(pt[7], dtype=np.float32)
                        norm_mag = np.linalg.norm(normal_grad)
                        if norm_mag > 1e-6:
                            normal_grad = normal_grad / norm_mag

                        # 3. Cartesian Velocity of Closest Link
                        link_a = pt[3]
                        fk_link_idx = -1 if (link_a < 0 or link_a >= 7) else (link_a + 1)
                        vel_cart = kin_helper.compute_cartesian_velocity(q_step, qd_step, link_idx=fk_link_idx)
                        vel_mag = np.linalg.norm(vel_cart)

                        # 4. Cosine Alignment theta
                        if vel_mag > 1e-4:
                            cos_theta = float(np.dot(normal_grad, vel_cart) / (norm_mag * vel_mag))
                        else:
                            cos_theta = 0.0

                        # 5. GVM-SDF Modulation Term
                        modulation = 1.0 - rho * cos_theta
                        gvm_term = phi * (1.0 + vel_mag * modulation)
                        step_coll_gvm += gvm_term

                coll_p[k, h] = step_coll_p
                coll_gvm[k, h] = step_coll_gvm

        # Interpolate across un-evaluated lookahead steps
        for h in range(H):
            if h not in eval_steps:
                prev_h = max([s for s in eval_steps if s <= h], default=0)
                coll_p[:, h] = coll_p[:, prev_h]
                coll_gvm[:, h] = coll_gvm[:, prev_h]

        return coll_p, coll_gvm

    def step_simulation(self):
        self.step()

    def reset(self):
        """Resets the Digital Twin for a new episode."""
        self.clear_dynamic_objects()
        if self.client_id >= 0 and self.table_id >= 0:
            try:
                p.removeBody(self.table_id)
            except Exception:
                pass
            self.table_id = -1

    def close(self):
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1

    def shutdown(self):
        self.close()

