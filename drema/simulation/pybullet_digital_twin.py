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
import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any


import pybullet as p
import pybullet_data
from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR, synchronized, aggregate_gvm_costs
from .mesh_decomposition import convex_decomposition
from drema.prediction import BaseObstaclePredictor, ObstacleTrajectoryPredictor
from drema.controller.franka_kinematics import FrankaKinematics
from scipy.spatial.transform import Rotation


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
        sim_substeps: int = 1,
        predictor: Optional[BaseObstaclePredictor] = None,
        joint_angle_offsets: Optional[List[float]] = None,
        table_in_collision_costs: bool = True,
        obstacle_convex_decomposition: bool = True,
        vhacd_resolution: int = 100000,
        vhacd_concavity: float = 0.05
    ):
        """
        :param obstacle_convex_decomposition: Dynamic PyBullet bodies collide through the convex hull of their mesh,
            which fills concave shapes (e.g. the free space under an arch). When True, obstacle meshes are split into
            convex parts with V-HACD (cached next to the mesh) and the compound is used for collision queries.
        :param vhacd_resolution: V-HACD voxel resolution.
        :param vhacd_concavity: V-HACD maximum concavity per part: higher values give fewer, coarser parts.
        """
        super().__init__(joint_angle_offsets)
        self.table_in_collision_costs = bool(table_in_collision_costs)
        self.obstacle_convex_decomposition = bool(obstacle_convex_decomposition)
        self.vhacd_resolution = int(vhacd_resolution)
        self.vhacd_concavity = float(vhacd_concavity)
        self.visualize = visualize
        self.table_z = table_z
        self.tracking_mode = tracking_mode.lower()
        self.constraint_max_force = float(constraint_max_force)
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot)
        self.kd_rot = float(kd_rot)
        self.sim_substeps = max(1, int(sim_substeps))

        # Modular generic trajectory predictor
        self.predictor: Optional[BaseObstaclePredictor] = (
            predictor if predictor is not None else ObstacleTrajectoryPredictor()
        )

        self.client_id = -1
        self.robot_id = -1
        self.robot_loaded = False
        self.table_id = -1

        # Dynamic registry of objects spawned by perception (obj_id -> object metadata)
        self.tracked_objects: Dict[int, Dict] = {}
        self.last_collision_timings: Dict[str, Any] = {}

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

    @synchronized
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
                self._reset_robot_base(base_position, base_orientation)
                if joint_positions is not None:
                    self.sync_robot_state(joint_positions)
                return True
            except Exception as e:
                print(f"[PYBULLET DIGITAL TWIN WARNING] Failed to update robot pose: {e}")
                return False

        # Load URDF at the exact base position received dynamically
        try:
            self.robot_id = p.loadURDF(self.robot_urdf_path, list(base_position), list(base_orientation), useFixedBase=True)
            self.robot_loaded = True
            self._open_fingers()
            print(f"[PYBULLET DIGITAL TWIN] Dynamically loaded Franka Panda URDF at [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}] (ID: {self.robot_id})")
        except Exception as e:
            print(f"[PYBULLET DIGITAL TWIN ERROR] Failed to load Franka Panda URDF: {e}")
            self.robot_id = -1
            self.robot_loaded = False
            return False

        if joint_positions is not None:
            self.sync_robot_state(joint_positions)

        return True

    def _open_fingers(self, opening: float = 0.04) -> None:
        """The URDF loads with the fingers closed; the twin keeps the gripper open like the controlled robot."""
        for j in range(p.getNumJoints(self.robot_id)):
            if b"finger_joint" in p.getJointInfo(self.robot_id, j)[1]:
                p.resetJointState(self.robot_id, j, opening)
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=opening)

    def _reset_robot_base(self, base_position, base_orientation) -> None:
        """
        Places the base link frame at the given pose. loadURDF takes the link frame, whereas
        resetBasePositionAndOrientation takes the base inertial frame, which for panda_link0 sits 5 cm
        above the link frame: passing the link pose directly would sink the whole robot by that offset.
        """
        local_pos, local_orn = p.getDynamicsInfo(self.robot_id, -1)[3:5]
        com_pos, com_orn = p.multiplyTransforms(list(base_position), list(base_orientation), local_pos, local_orn)
        p.resetBasePositionAndOrientation(self.robot_id, com_pos, com_orn)

    @synchronized
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
                self._reset_robot_base(base_position, base_orientation)
            except Exception as e:
                print(f"[PYBULLET DIGITAL TWIN WARNING] Failed to reset robot base pose: {e}")

    @synchronized
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

    def _collision_mesh_path(self, mesh_path: str) -> str:
        """Mesh used for collision queries: the cached V-HACD convex decomposition, or the mesh itself."""
        if not self.obstacle_convex_decomposition:
            return mesh_path
        try:
            return convex_decomposition(mesh_path, self.vhacd_resolution, self.vhacd_concavity)
        except Exception as e:
            print(f"[PYBULLET DIGITAL TWIN WARNING] Convex decomposition failed ({e}): using the convex hull of '{mesh_path}'")
            return mesh_path

    @synchronized
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

            col_id = p.createCollisionShape(p.GEOM_MESH, fileName=self._collision_mesh_path(mesh_path))
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
                'target_quat': initial_quat,
                'is_target': bool(kwargs.get('is_target', False))
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

    @synchronized
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

    @synchronized
    def clear_dynamic_objects(self):
        """Removes all dynamically discovered objects upon reset."""
        for obj_id in list(self.tracked_objects.keys()):
            self.remove_object(obj_id)
        self.tracked_objects.clear()

    # -------------------------------------------------------------------------
    # State Synchronization & Physics Queries
    # -------------------------------------------------------------------------

    @synchronized
    def sync_robot_state(self, joint_positions: List[float]):
        """Synchronizes robot joint angles in the Digital Twin from CoppeliaSim."""
        if self.robot_id < 0:
            return

        num_joints = p.getNumJoints(self.robot_id)
        arm_j = 0
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED and arm_j < len(joint_positions):
                offset = self.joint_angle_offsets[arm_j] if arm_j < 7 else 0.0
                p.resetJointState(self.robot_id, i, float(joint_positions[arm_j]) + offset)
                arm_j += 1

        if self.attachments and len(joint_positions) >= 7:
            for obj_id, (pos, quat) in self.attached_poses(np.asarray(joint_positions[:7])).items():
                self._place_held_object(obj_id, pos[0], quat[0], persistent=True)

    # PyBullet link indices >= 7 are the flange, hand, fingers and grasp target, all rigid with link7.
    HAND_LINK_MIN = 7

    def _place_held_object(self, obj_id: int, pos: np.ndarray, quat_xyzw: np.ndarray, persistent: bool) -> None:
        """Puts a held object at its TCP-attached pose; persistent also moves its tracking target."""
        obj = self.tracked_objects.get(obj_id)
        if obj is None:
            return
        pos_l = [float(x) for x in pos]
        quat_l = [float(x) for x in quat_xyzw]
        p.resetBasePositionAndOrientation(obj['body_id'], pos_l, quat_l)
        if persistent:
            obj['target_pos'] = tuple(pos_l)
            obj['target_quat'] = tuple(quat_l)
            cid = obj.get('constraint_id', -1)
            if cid >= 0:
                p.changeConstraint(cid, jointChildPivot=pos_l, jointChildFrameOrientation=quat_l,
                                   maxForce=self.constraint_max_force)

    def _set_robot_collision_with(self, body_id: int, enable: bool) -> None:
        for link in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(self.robot_id, body_id, link, -1, int(enable))

    def _on_attach(self, obj_id: int) -> None:
        if self.robot_id >= 0 and obj_id in self.tracked_objects:
            self._set_robot_collision_with(self.tracked_objects[obj_id]['body_id'], False)

    def _on_detach(self, obj_id: int) -> None:
        if self.robot_id >= 0 and obj_id in self.tracked_objects:
            self._set_robot_collision_with(self.tracked_objects[obj_id]['body_id'], True)

    @synchronized
    def sync_gripper(self, open_amount: float) -> None:
        if self.robot_id >= 0:
            self._open_fingers(0.04 * float(np.clip(open_amount, 0.0, 1.0)))

    def _collision_pairs(self, include_table: bool):
        """
        Bodies the robot must keep clear, according to the object policies:
        - obstacles: (body_id, obj_id or -1 for the table, policy);
        - held: [(body_id, obj_id)] held objects, checked as part of the hand.
        """
        obstacles, held = [], []
        for obj_id, obj in self.tracked_objects.items():
            body_id = obj.get('body_id', -1)
            if body_id < 0:
                continue
            if obj_id in self.attachments:
                held.append((body_id, obj_id))
                continue
            policy = self.object_policy(obj_id)
            if policy.hand_contact and policy.arm_contact:
                continue
            obstacles.append((body_id, obj_id, policy))
        if include_table and self.table_id >= 0:
            obstacles.append((self.table_id, -1, self.table_policy))
        return obstacles, held

    def _robot_link_allowed(self, policy, link_a: int) -> bool:
        return policy.pair_allowed(link_a >= self.HAND_LINK_MIN)

    @staticmethod
    def _query_margin(policy) -> float:
        """Largest distance offset over the hand and arm pairs of an object (the closest-point query radius)."""
        margins = [m for m in (policy.pair_margin(True), policy.pair_margin(False)) if m is not None]
        return max(margins) if margins else 0.0

    @synchronized
    def sync_object_pose(
        self,
        obj_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        timestamp: Optional[float] = None
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

        # A held object follows the TCP; perception (often occluded by the fingers) is kept for diagnostics only.
        if obj_id in self.attachments:
            obj_data['perceived_pos'] = pos
            return

        obj_data['target_pos'] = pos
        obj_data['target_quat'] = orn

        # Pass 6D pose to generic trajectory predictor
        if self.predictor is not None:
            self.predictor.update_obstacle_pose(obj_id, pos, orn, timestamp=timestamp)

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

    @synchronized
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

    @synchronized
    def tracked_object_pose(self, obj_id: int):
        # In constraint / PD mode the body only reaches its tracked pose after the next physics steps.
        obj = self.tracked_objects.get(obj_id)
        if obj is None or 'target_pos' not in obj:
            return None
        return tuple(float(x) for x in obj['target_pos']), tuple(float(x) for x in obj.get('target_quat', (0, 0, 0, 1)))

    @synchronized
    def get_min_obstacle_distance(self) -> float:
        """
        Calculates the minimum clearance distance between the Franka Panda arm and
        ANY currently tracked dynamic obstacle in the scene.
        Used by the MPC controller for proactive collision avoidance.
        """
        if self.robot_id < 0 or len(self.tracked_objects) == 0:
            return float('inf')

        min_distance = float('inf')
        obstacles, held = self._collision_pairs(include_table=False)
        for body_id, obj_id, policy in obstacles:
            for pt in p.getClosestPoints(bodyA=self.robot_id, bodyB=body_id, distance=2.0):
                if self._robot_link_allowed(policy, pt[3]):
                    min_distance = min(min_distance, pt[8])
            if policy.pair_allowed(True):
                for held_body, _ in held:
                    for pt in p.getClosestPoints(bodyA=held_body, bodyB=body_id, distance=2.0):
                        min_distance = min(min_distance, pt[8])

        return float(min_distance)

    @synchronized
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

    @synchronized
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
                vel = (0.0, 0.0, 0.0)
                if self.predictor is not None:
                    st = self.predictor.get_estimated_state(obj_id)
                    if st is not None:
                        vel = tuple(float(x) for x in st['velocity'])

                obstacles.append({
                    'id': obj_id,
                    'body_id': body_id,
                    'name': obj_data.get('name', f"obstacle_{obj_id}"),
                    'position': pos,
                    'orientation': orn if orn is not None else (0.0, 0.0, 0.0, 1.0),
                    'velocity': vel,
                    'is_target': bool(obj_data.get('is_target', False))
                })
        return obstacles

    @synchronized
    def get_obstacle_proximity(self, max_distance: float) -> List[Dict[str, Any]]:
        """Closest robot/obstacle surface points per (obstacle, link) pair, via PyBullet getClosestPoints."""
        if self.client_id < 0 or self.robot_id < 0:
            return []

        entries = []
        obstacles, held = self._collision_pairs(include_table=False)
        for body_id, obj_id, policy in obstacles:
            velocity = (0.0, 0.0, 0.0)
            if self.predictor is not None:
                st = self.predictor.get_estimated_state(obj_id)
                if st is not None:
                    velocity = tuple(float(x) for x in st['velocity'])

            query = max(float(max_distance) + self._query_margin(policy), 1e-4)
            # link_idx -> (closest point, distance shifted by the pair margin)
            closest_per_link: Dict[int, Any] = {}
            for pt in p.getClosestPoints(bodyA=self.robot_id, bodyB=body_id, distance=query):
                # PyBullet link -1 is the fixed base (link0); 0..6 are link1..link7; flange, hand,
                # fingers and grasp target are rigidly attached to link7.
                margin = policy.pair_margin(pt[3] >= self.HAND_LINK_MIN)
                if pt[3] < 0 or margin is None:
                    continue
                link_idx = min(pt[3] + 1, 7)
                d = float(pt[8]) - margin
                if d < max_distance and (link_idx not in closest_per_link or d < closest_per_link[link_idx][1]):
                    closest_per_link[link_idx] = (pt, d)
            hand_margin = policy.pair_margin(True)
            if hand_margin is not None:
                # Held objects move with link7: their closest points count as link7 points.
                for held_body, _ in held:
                    for pt in p.getClosestPoints(bodyA=held_body, bodyB=body_id, distance=query):
                        d = float(pt[8]) - hand_margin
                        if d < max_distance and (7 not in closest_per_link or d < closest_per_link[7][1]):
                            closest_per_link[7] = (pt, d)

            for link_idx, (pt, d) in closest_per_link.items():
                normal = np.array(pt[7], dtype=np.float64)
                entries.append({
                    'obj_id': obj_id,
                    'link_idx': link_idx,
                    'point_world': np.array(pt[5], dtype=np.float64),
                    'normal': normal / max(np.linalg.norm(normal), 1e-9),
                    'distance': d,
                    'velocity': velocity
                })
        return entries

    @synchronized
    def calculate_inverse_kinematics(
        self,
        target_pos: Tuple[float, float, float],
        target_quat: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        Calculates inverse kinematics solution using PyBullet C++ solver with Franka joint limits.
        The solved frame is the URDF panda_grasptarget, which is rotated by -45 deg about z with
        respect to the FrankaKinematics EE frame; the MPC therefore uses FrankaKinematics.solve_dls_ik.
        """
        if self.client_id < 0 or self.robot_id < 0:
            return None
        try:
            ik_target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
            kwargs = {
                "maxNumIterations": 50,
                "residualThreshold": 1e-4
            }
            if target_quat is not None:
                tq = np.asarray(target_quat)
                if tq.shape == (3, 3):
                    kwargs["targetOrientation"] = [float(x) for x in Rotation.from_matrix(tq).as_quat()]
                elif len(tq.flatten()) == 4:
                    kwargs["targetOrientation"] = [float(x) for x in tq.flatten()]

            num_j = p.getNumJoints(self.robot_id)
            ee_idx = 11 if num_j > 11 else max(0, num_j - 1)

            # Supply Franka Panda joint limits and rest poses (expressed in the twin's joint space).
            # PyBullet only enables null-space IK when these lists have one entry per movable DoF
            # (7 arm joints + 2 fingers); with any other length they are silently ignored.
            offsets = self.joint_angle_offsets
            q_min_7 = (FrankaKinematics.Q_MIN + offsets).tolist()
            q_max_7 = (FrankaKinematics.Q_MAX + offsets).tolist()
            rest_7 = (FrankaKinematics.Q_REST + offsets).tolist()
            num_dofs = sum(1 for j in range(num_j) if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED)
            extra_j = max(0, num_dofs - 7)

            q_min = q_min_7 + [0.0] * extra_j
            q_max = q_max_7 + [0.04] * extra_j
            q_range = [mx - mn for mn, mx in zip(q_min, q_max)]
            rest_poses = rest_7 + [0.0] * extra_j

            kwargs["lowerLimits"] = q_min
            kwargs["upperLimits"] = q_max
            kwargs["jointRanges"] = q_range
            kwargs["restPoses"] = rest_poses

            pb_ik = p.calculateInverseKinematics(
                self.robot_id,
                ee_idx,
                ik_target,
                **kwargs
            )
            q_cand = (np.array(pb_ik[:7], dtype=np.float64) - offsets).astype(np.float32)

            # Verify that candidate strictly satisfies physical joint limits
            if np.any(q_cand < (FrankaKinematics.Q_MIN - 1e-3)) or np.any(q_cand > (FrankaKinematics.Q_MAX + 1e-3)):
                return None
            return q_cand
        except Exception:
            return None

    @synchronized
    def compute_trajectory_collision_costs(
        self,
        Q: np.ndarray,
        QD: np.ndarray,
        sigma_1: float,
        sigma_2: float,
        kappa: float,
        rho: float,
        kin_helper: Any,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates collision clearance and GVM (Gradient-Velocity Modulated) costs for
        a batch of candidate trajectories Q (K x H x 7) and QD (K x H x 7) using PyBullet GJK/EPA.
        Obstacle future states are dynamically forecasted across horizon H via self.predictor.
        """
        K, H, _ = Q.shape
        coll_p = np.zeros((K, H), dtype=np.float32)
        coll_gvm = np.zeros((K, H), dtype=np.float32)

        if self.client_id < 0 or self.robot_id < 0:
            self.last_collision_timings = {
                'backend': 'PyBullet',
                'pred_ms': 0.0,
                'backend_ms': 0.0,
                'bullet_ms': 0.0,
                'gvm_ms': 0.0,
                'pts_count': 0
            }
            return coll_p, coll_gvm

        # Obstacles allowed by the object policies (body_id -> obj_id for the tracked ones). The table is static
        # (no prediction, zero velocity); only the moving links can be penalized against it, since the fixed base
        # link rests on its surface. Held objects move with the TCP and are checked as part of the hand.
        obstacles, held = self._collision_pairs(include_table=self.table_in_collision_costs)
        obstacle_body_ids: List[int] = [b for b, _, _ in obstacles]
        obstacle_policies = [pol for _, _, pol in obstacles]
        body_to_obj_id: Dict[int, int] = {b: o for b, o, _ in obstacles if o >= 0}

        if not obstacle_body_ids:
            self.last_collision_timings = {
                'backend': 'PyBullet',
                'pred_ms': 0.0,
                'backend_ms': 0.0,
                'bullet_ms': 0.0,
                'gvm_ms': 0.0,
                'pts_count': 0
            }
            return coll_p, coll_gvm

        robot_id = self.robot_id

        # Forecast future trajectories for all tracked dynamic obstacles across horizon H
        t_pred_start = time.perf_counter()
        predictions = (
            self.predictor.predict_all(horizon=H, dt=dt)
            if self.predictor is not None
            else {}
        )
        t_pred_ms = (time.perf_counter() - t_pred_start) * 1000.0

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
        saved_arm_states = [p.getJointState(robot_id, j_idx)[:2] for j_idx in arm_joint_indices]

        t_bullet_total = 0.0
        total_contact_pts = 0

        # Closest-point records, aggregated per (obstacle, robot link) as in Zhou et al. Eq. 10, 21:
        # the potential is evaluated per kinematic link instead of summing over dense mesh vertices.
        # Each held object adds one extra "link" group after the robot links.
        n_links = num_j + len(held)
        n_eval = len(eval_steps)
        held_poses = self.attached_poses(Q[:, eval_steps].reshape(-1, 7)) if held else {}
        rec_k: List[int] = []
        rec_e: List[int] = []
        rec_group: List[int] = []
        rec_slot: List[int] = []
        rec_d: List[float] = []
        rec_normal: List[Any] = []
        rec_v_obs: List[Any] = []
        rec_point: List[Any] = []
        record_points = self.record_collision_points
        zero_vel = (0.0, 0.0, 0.0)

        for e, h in enumerate(eval_steps):
            # Advance all tracked dynamic obstacles to their forecasted future pose at lookahead step h
            for b_id, o_id in body_to_obj_id.items():
                if o_id in predictions:
                    pred = predictions[o_id]
                    p.resetBasePositionAndOrientation(
                        b_id,
                        pred.positions[h].tolist(),
                        pred.orientations[h].tolist()
                    )
            obs_vel = []
            for obs_id in obstacle_body_ids:
                o_id = body_to_obj_id.get(obs_id)
                obs_vel.append(predictions[o_id].velocities[h] if o_id is not None and o_id in predictions else zero_vel)

            t_b_start = time.perf_counter()
            for k in range(K):
                q_step = Q[k, h]
                for arm_j, j_idx in enumerate(arm_joint_indices):
                    p.resetJointState(robot_id, j_idx, float(q_step[arm_j]) + self.joint_angle_offsets[arm_j])
                for held_body, held_obj in held:
                    pos_h, quat_h = held_poses[held_obj]
                    self._place_held_object(held_obj, pos_h[k * n_eval + e], quat_h[k * n_eval + e], persistent=False)

                for slot, obs_id in enumerate(obstacle_body_ids):
                    policy = obstacle_policies[slot]
                    query = max(float(sigma_2) + self._query_margin(policy), 1e-4)
                    hand_margin = policy.pair_margin(True)
                    # (closest point, group link, velocity slot, pair margin): robot links, then held objects as hand parts
                    candidates = []
                    for pt in p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=query):
                        margin = policy.pair_margin(pt[3] >= self.HAND_LINK_MIN)
                        # fixed base link (link0): its distance cannot be changed by any control
                        if pt[3] >= 0 and margin is not None:
                            candidates.append((pt, pt[3], pt[3] if pt[3] < 7 else 7, margin))
                    if hand_margin is not None:
                        for h_idx, (held_body, _) in enumerate(held):
                            candidates.extend(
                                (pt, num_j + h_idx, 7, hand_margin)
                                for pt in p.getClosestPoints(bodyA=held_body, bodyB=obs_id, distance=query))
                    total_contact_pts += len(candidates)
                    for pt, group_link, vel_slot, margin in candidates:
                        # Per-pair margins shift the distance: FRAGILE objects farther, approach keyframes closer to the hand.
                        if pt[8] - margin >= sigma_2:
                            continue
                        rec_k.append(k)
                        rec_e.append(e)
                        rec_group.append(slot * n_links + group_link)
                        rec_slot.append(vel_slot)
                        rec_d.append(pt[8] - margin)
                        rec_normal.append(pt[7])
                        rec_v_obs.append(obs_vel[slot])
                        if record_points:
                            rec_point.append(pt[5])
            t_bullet_total += time.perf_counter() - t_b_start

        if record_points:
            slot_obj = np.array([body_to_obj_id.get(b, -1) for b in obstacle_body_ids], dtype=np.int64)
            self.last_collision_points = {
                'rollout': np.asarray(rec_k, dtype=np.int64),
                'step': np.asarray(eval_steps, dtype=np.int64)[np.asarray(rec_e, dtype=np.int64)],
                'position': np.asarray(rec_point, dtype=np.float64).reshape(-1, 3),
                'distance': np.asarray(rec_d, dtype=np.float64),
                'obstacle': slot_obj[np.asarray(rec_group, dtype=np.int64) // n_links] if rec_group else np.zeros(0, np.int64),
            }

        t_gvm_start = time.perf_counter()
        coll_p_eval, coll_gvm_eval = aggregate_gvm_costs(
            Q[:, eval_steps], QD[:, eval_steps], rec_k, rec_e, rec_group, rec_slot, rec_d, rec_normal, rec_v_obs,
            n_groups=len(obstacle_body_ids) * n_links, sigma_1=sigma_1, sigma_2=sigma_2, kappa=kappa, rho=rho,
            kin_helper=kin_helper
        )
        coll_p[:, eval_steps] = coll_p_eval
        coll_gvm[:, eval_steps] = coll_gvm_eval
        t_gvm_total = time.perf_counter() - t_gvm_start

        self.last_collision_timings = {
            'backend': 'PyBullet',
            'pred_ms': t_pred_ms,
            'backend_ms': t_bullet_total * 1000.0,
            'bullet_ms': t_bullet_total * 1000.0,
            'gvm_ms': t_gvm_total * 1000.0,
            'pts_count': total_contact_pts
        }

        for j_idx, (q_saved, qd_saved) in zip(arm_joint_indices, saved_arm_states):
            p.resetJointState(robot_id, j_idx, q_saved, qd_saved)

        # Restore all dynamic obstacles (and held objects) to their current instantaneous state (t=0)
        for b_id, o_id in list(body_to_obj_id.items()) + held:
            if 'target_pos' in self.tracked_objects[o_id]:
                p.resetBasePositionAndOrientation(
                    b_id,
                    list(self.tracked_objects[o_id]['target_pos']),
                    list(self.tracked_objects[o_id].get('target_quat', (0, 0, 0, 1)))
                )

        # Interpolate across un-evaluated lookahead steps
        for h in range(H):
            if h not in eval_steps:
                prev_h = max([s for s in eval_steps if s <= h], default=0)
                coll_p[:, h] = coll_p[:, prev_h]
                coll_gvm[:, h] = coll_gvm[:, prev_h]

        return coll_p, coll_gvm

    def step_simulation(self):
        self.step()

    @synchronized
    def reset(self):
        """Resets the Digital Twin for a new episode."""
        self.attachments.clear()
        self.object_policies = {}
        self.clear_dynamic_objects()
        if self.predictor is not None:
            self.predictor.reset()
        if self.client_id >= 0 and self.table_id >= 0:
            try:
                p.removeBody(self.table_id)
            except Exception:
                pass
            self.table_id = -1

    @synchronized
    def close(self):
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1

    def shutdown(self):
        self.close()

