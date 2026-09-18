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

import pybullet as p
import pybullet_data


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
        self.robot_id = -1

    def load_robot(
        self,
        base_position: Tuple[float, float, float],
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        joint_positions: Optional[List[float]] = None
    ) -> int:
        """
        Dynamically loads or updates Franka Panda robot in PyBullet at the given world position.
        """
        if self.client_id < 0:
            return -1

        if self.robot_id >= 0:
            try:
                p.resetBasePositionAndOrientation(self.robot_id, list(base_position), list(base_orientation))
                if joint_positions is not None:
                    self.sync_robot_state(joint_positions)
                return self.robot_id
            except Exception as e:
                print(f"[DigitalTwin Warning] Failed to update robot pose: {e}")

        # Load URDF at the exact base position received dynamically
        try:
            self.robot_id = p.loadURDF(self.robot_urdf_path, list(base_position), list(base_orientation), useFixedBase=True)
            print(f"✓ Digital Twin: Dynamically loaded Franka Panda URDF at [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}] (ID: {self.robot_id})")
        except Exception as e:
            print(f"[DigitalTwin Warning] Failed to load Franka Panda URDF: {e}")
            return -1

        if joint_positions is not None:
            self.sync_robot_state(joint_positions)

        return self.robot_id

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
                print(f"[DigitalTwin Warning] Failed to reset robot base pose: {e}")

    def get_robot_link_positions(self) -> List[Tuple[float, float, float]]:
        """Returns the 3D world positions of all Franka robot links in PyBullet."""
        if self.client_id < 0 or self.robot_id < 0:
            return []
        try:
            positions = [p.getBasePositionAndOrientation(self.robot_id)[0]]
            num_joints = p.getNumJoints(self.robot_id)
            for i in range(num_joints):
                state = p.getLinkState(self.robot_id, i)
                positions.append(state[0])
            return positions
        except Exception:
            return []

    def get_dense_robot_skeleton_points(self, num_samples_per_link: int = 4) -> List[Tuple[float, float, float]]:
        """
        Returns dense 3D points sampled along Franka robot link segments for robust pointcloud/depth masking.
        """
        if self.client_id < 0 or self.robot_id < 0:
            return []
        try:
            link_pos = self.get_robot_link_positions()
            if len(link_pos) <= 1:
                return link_pos
            dense_points = [link_pos[0]]
            for i in range(len(link_pos) - 1):
                p1 = np.array(link_pos[i], dtype=np.float32)
                p2 = np.array(link_pos[i + 1], dtype=np.float32)
                alphas = np.linspace(0.0, 1.0, num_samples_per_link + 2)[1:]
                for a in alphas:
                    pt = (1.0 - a) * p1 + a * p2
                    dense_points.append(tuple(pt.tolist()))
            return dense_points
        except Exception:
            return self.get_robot_link_positions()

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

    def draw_voxel_grid_bbox(
        self,
        origin: Tuple[float, float, float],
        dim: Tuple[int, int, int],
        voxel_size: float,
        color: Tuple[float, float, float] = (0.0, 0.9, 0.25),
        line_width: float = 2.0
    ):
        """
        Draws the 12 wireframe edges and center coordinate cross of the TSDF Voxel Grid in PyBullet GUI.
        """
        if self.client_id < 0:
            return

        x0, y0, z0 = origin
        x1 = x0 + dim[0] * voxel_size
        y1 = y0 + dim[1] * voxel_size
        z1 = z0 + dim[2] * voxel_size

        edges = [
            # Bottom 4 edges
            ([x0, y0, z0], [x1, y0, z0]), ([x1, y0, z0], [x1, y1, z0]),
            ([x1, y1, z0], [x0, y1, z0]), ([x0, y1, z0], [x0, y0, z0]),
            # Top 4 edges
            ([x0, y0, z1], [x1, y0, z1]), ([x1, y0, z1], [x1, y1, z1]),
            ([x1, y1, z1], [x0, y1, z1]), ([x0, y1, z1], [x0, y0, z1]),
            # 4 Vertical edges
            ([x0, y0, z0], [x0, y0, z1]), ([x1, y0, z0], [x1, y0, z1]),
            ([x1, y1, z0], [x1, y1, z1]), ([x0, y1, z0], [x0, y1, z1]),
        ]
        for p0, p1 in edges:
            p.addUserDebugLine(p0, p1, lineColorRGB=list(color), lineWidth=line_width)

        # Center marker coordinate cross
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        cz = (z0 + z1) / 2.0
        d = 0.06
        p.addUserDebugLine([cx - d, cy, cz], [cx + d, cy, cz], lineColorRGB=[1, 0, 0], lineWidth=4)
        p.addUserDebugLine([cx, cy - d, cz], [cx, cy + d, cz], lineColorRGB=[0, 1, 0], lineWidth=4)
        p.addUserDebugLine([cx, cy, cz - d], [cx, cy, cz + d], lineColorRGB=[0, 0, 1], lineWidth=4)

        # 3D Text Label in PyBullet Scene
        p.addUserDebugText(
            f"Voxel Grid: {dim[0]}x{dim[1]}x{dim[2]} ({voxel_size*100:.1f}cm)\nCenter: [{cx:.2f}, {cy:.2f}, {cz:.2f}]",
            [cx - 0.15, cy, z1 + 0.03],
            textColorRGB=[0.0, 1.0, 0.4],
            textSize=1.1,
            lifeTime=0
        )
        print(f"✓ Digital Twin: Voxel Grid wireframe bbox rendered in PyBullet (Center: [{cx:.3f}, {cy:.3f}, {cz:.3f}])")

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
        if self.client_id < 0:
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
        if self.client_id < 0:
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
        if self.client_id < 0:
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
        """
        if obj_id not in self.tracked_objects:
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
        if self.robot_id < 0 or len(self.tracked_objects) == 0:
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
        if self.client_id >= 0:
            p.stepSimulation()

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
