#!/usr/bin/env python
"""
PyBullet Digital Twin for DREMA Dynamic Inference Suite.

Agnostic & Dynamic Scene Manager:
- Only the static workcell baseline is loaded at startup (Ground plane, Table workspace, Franka Panda robot).
- All scene objects (obstacles, containers, manipulated objects, target) are dynamically spawned
  and updated at runtime as they are discovered and reconstructed by the VG-Mapping pipeline
  from point clouds and 3D surface meshes.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, List, Dict, Union

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    p = None
    pybullet_data = None
    HAS_PYBULLET = False


class PyBulletDigitalTwin:
    """
    Physical Digital Twin of the manipulation scene running in PyBullet.
    """

    def __init__(
        self,
        visualize: bool = False,
        table_z: float = 0.75,
        robot_urdf_path: Optional[str] = None
    ):
        self.visualize = visualize
        self.table_z = table_z

        self.client_id = -1
        self.robot_id = -1
        self.table_id = -1
        self.target_body_id = -1

        # Dynamic registry of objects spawned by perception (obj_id -> object metadata)
        self.tracked_objects: Dict[int, Dict] = {}

        if not HAS_PYBULLET:
            print("[DigitalTwin Warning] PyBullet is not installed. Running in dummy mode.")
            return

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

        # 2. Known Baseline Model: Franka Panda Arm URDF
        if robot_urdf_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            robot_urdf_path = os.path.join(base_dir, "assets/franka_panda/panda.urdf")

        if os.path.exists(robot_urdf_path):
            self.robot_id = p.loadURDF(robot_urdf_path, [0.0, 0.0, 0.0], [0, 0, 0, 1], useFixedBase=True)
            print(f"✓ Digital Twin: Loaded Franka Panda robot URDF (ID: {self.robot_id})")
        else:
            try:
                self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], [0, 0, 0, 1], useFixedBase=True)
                print(f"✓ Digital Twin: Loaded fallback PyBullet panda.urdf (ID: {self.robot_id})")
            except Exception as e:
                print(f"[DigitalTwin Warning] Failed to load Franka Panda URDF: {e}")

    def spawn_scanned_table(
        self,
        table_z: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        mesh_file_path: Optional[str] = None
    ) -> int:
        """
        Dynamically spawns the tabletop workspace surface from real 3D scanning data at t=0.
        Does NOT rely on hardcoded table geometry!
        """
        if not HAS_PYBULLET or self.client_id < 0:
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
                vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file_path, rgbaColor=[0.75, 0.75, 0.75, 1.0])
                self.table_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[0, 0, 0]
                )
                print(f"✓ Digital Twin: Spawned scanned table from 3D surface mesh: {mesh_file_path} (ID: {self.table_id})")
                return self.table_id
            except Exception as e:
                print(f"[DigitalTwin Warning] Failed to load table mesh {mesh_file_path}: {e}. Falling back to planar bounds.")

        # Spawn table structure extending from ground Z=0 to surface Z=table_z
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            hx = max(0.35, (xmax - xmin) / 2.0)
            hy = max(0.35, (ymax - ymin) / 2.0)
        else:
            cx, cy = 0.30, 0.0
            hx, hy = 0.80, 0.55

        # Table body reaches from Z=0 to table_z
        hz = self.table_z / 2.0
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.82, 0.82, 0.82, 1.0])
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[cx, cy, hz]
        )
        print(f"✓ Digital Twin: Spawned scanned table structure with bounds X[{cx-hx:.2f}, {cx+hx:.2f}], Y[{cy-hy:.2f}, {cy+hy:.2f}], Z=[0.0, {self.table_z:.3f}]m (ID: {self.table_id})")
        return self.table_id

    # -------------------------------------------------------------------------
    # Generic Dynamic Object Spawning (Called by VG-Mapping pipeline)
    # -------------------------------------------------------------------------

    def spawn_mesh_object(
        self,
        obj_id: int,
        mesh_file_path: str,
        initial_position: Tuple[float, float, float],
        initial_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        color: Tuple[float, float, float, float] = (0.2, 0.45, 0.85, 1.0),
        is_target: bool = False
    ) -> int:
        """
        Dynamically inserts an object into the Digital Twin from a 3D surface mesh
        extracted by VG-Mapping at t=0.
        """
        if not HAS_PYBULLET or self.client_id < 0:
            return -1

        # If object was already spawned, remove old body first
        if obj_id in self.tracked_objects:
            self.remove_object(obj_id)

        try:
            col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file_path)
            vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file_path, rgbaColor=list(color))
            body_id = p.createMultiBody(
                baseMass=0.1 if not is_target else 0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=list(initial_position),
                baseOrientation=list(initial_orientation)
            )

            self.tracked_objects[obj_id] = {
                'body_id': body_id,
                'mesh_path': mesh_file_path,
                'is_target': is_target,
                'color': color,
                'canonical_position': initial_position
            }
            if is_target:
                self.target_body_id = body_id

            print(f"✓ Digital Twin: Spawned mesh object ID {obj_id} (PyBullet Body ID: {body_id}, Target={is_target})")
            return body_id
        except Exception as e:
            print(f"[DigitalTwin Error] Failed to spawn mesh object {obj_id}: {e}")
            return -1

    def spawn_box_object(
        self,
        obj_id: int,
        half_extents: Tuple[float, float, float],
        initial_position: Tuple[float, float, float],
        initial_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        is_target: bool = False
    ) -> int:
        """
        Dynamically inserts a bounding primitive object into the Digital Twin.
        """
        if not HAS_PYBULLET or self.client_id < 0:
            return -1

        if obj_id in self.tracked_objects:
            self.remove_object(obj_id)

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(color))
        body_id = p.createMultiBody(
            baseMass=0.1 if not is_target else 0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(initial_position),
            baseOrientation=list(initial_orientation)
        )

        self.tracked_objects[obj_id] = {
            'body_id': body_id,
            'half_extents': half_extents,
            'is_target': is_target,
            'color': color,
            'canonical_position': initial_position
        }
        if is_target:
            self.target_body_id = body_id

        return body_id

    def remove_object(self, obj_id: int):
        """Removes an object from PyBullet."""
        if not HAS_PYBULLET or self.client_id < 0:
            return
        if obj_id in self.tracked_objects:
            body_id = self.tracked_objects[obj_id]['body_id']
            p.removeBody(body_id)
            if self.target_body_id == body_id:
                self.target_body_id = -1
            del self.tracked_objects[obj_id]

    def clear_dynamic_objects(self):
        """Removes all dynamically discovered objects upon reset."""
        for obj_id in list(self.tracked_objects.keys()):
            self.remove_object(obj_id)
        self.tracked_objects.clear()
        self.target_body_id = -1

    # -------------------------------------------------------------------------
    # State Synchronization & Physics Queries
    # -------------------------------------------------------------------------

    def sync_robot_state(self, joint_positions: List[float]):
        """Synchronizes robot joint angles in the Digital Twin from CoppeliaSim."""
        if not HAS_PYBULLET or self.robot_id < 0:
            return

        num_movable = min(len(joint_positions), p.getNumJoints(self.robot_id))
        for j_idx in range(num_movable):
            p.resetJointState(self.robot_id, j_idx, joint_positions[j_idx])

    def sync_object_pose(
        self,
        obj_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float]
    ):
        """
        Synchronizes the 3D position and orientation of an object tracked by RecurGS SE(3).
        """
        if not HAS_PYBULLET or obj_id not in self.tracked_objects:
            return

        body_id = self.tracked_objects[obj_id]['body_id']

        # Enforce table support boundary
        pos_z = max(position[2], self.table_z + 0.005)
        pos = (float(position[0]), float(position[1]), float(pos_z))
        p.resetBasePositionAndOrientation(body_id, pos, orientation)

    def get_min_obstacle_distance(self) -> float:
        """
        Calculates the minimum clearance distance between the Franka Panda arm and
        ANY currently tracked dynamic obstacle in the scene.
        Used by the MPC controller for proactive collision avoidance.
        """
        if not HAS_PYBULLET or self.robot_id < 0 or len(self.tracked_objects) == 0:
            return float('inf')

        min_distance = float('inf')

        for obj_id, obj_data in self.tracked_objects.items():
            # Only evaluate collisions against obstacles (ignore the target to reach)
            if obj_data.get('is_target', False):
                continue

            body_id = obj_data['body_id']
            contact_pts = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=body_id,
                distance=0.5
            )

            if contact_pts:
                obj_min_d = min(pt[8] for pt in contact_pts)
                if obj_min_d < min_distance:
                    min_distance = obj_min_d

        return float(min_distance)

    def step(self):
        if HAS_PYBULLET and self.client_id >= 0:
            p.stepSimulation()

    def reset(self):
        """Resets the Digital Twin for a new episode."""
        self.clear_dynamic_objects()
        if HAS_PYBULLET and self.client_id >= 0 and self.table_id >= 0:
            try:
                p.removeBody(self.table_id)
            except Exception:
                pass
            self.table_id = -1

    def close(self):
        if HAS_PYBULLET and self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1
