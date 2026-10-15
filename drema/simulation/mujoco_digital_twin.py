#!/usr/bin/env python
"""
MuJoCo Digital Twin for DREMA Dynamic Inference System.

Agnostic & Dynamic Scene Manager running on MuJoCo:
- Loads the Franka Emika Panda from the MuJoCo Menagerie MJCF (collision geometry only) and the static
  environment (ground, table).
- Dynamically compiles discovered 3D surface mesh obstacles extracted by Perception.
- Fluid dynamic multi-body tracking using MuJoCo Mocap Bodies + Weld Equality Constraints
  (or Cartesian PD force/torque with gravity feedforward), natively enforcing non-interpenetration.
- High-performance C physics (<15 microseconds/step), ready for GPU batching with MJX / Warp.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any
from scipy.spatial.transform import Rotation

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR, synchronized, aggregate_gvm_costs
from .mesh_decomposition import convex_parts
from drema.controller.franka_kinematics import FrankaKinematics
from drema.prediction import BaseObstaclePredictor, ObstacleTrajectoryPredictor


def _xyzw_to_wxyz(q: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    """Converts ROS/PyBullet quaternion [x, y, z, w] to MuJoCo quaternion [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _wxyz_to_xyzw(q: Union[Tuple[float, ...], List[float], np.ndarray]) -> Tuple[float, float, float, float]:
    """Converts MuJoCo quaternion [w, x, y, z] to ROS/PyBullet quaternion [x, y, z, w]."""
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


def _matrix_to_quat_xyzw(rot_mat: np.ndarray) -> Tuple[float, float, float, float]:
    """Converts 3x3 rotation matrix to quaternion [x, y, z, w]."""
    q_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_wxyz, rot_mat.astype(np.float64).flatten())
    return _wxyz_to_xyzw(q_wxyz)


DEFAULT_ROBOT_MODEL = "assets/mujoco_menagerie/franka_emika_panda/panda_upstream.xml"
# Robot bodies whose geometry follows the TCP (link index 8 in the robot geom map)
HAND_BODIES = ("hand", "left_finger", "right_finger")
FINGER_OPENING = 0.04  # [m] per finger: the twin keeps the gripper open


def _world_aabbs(model, data, geom_ids: np.ndarray):
    """World-frame AABB centres and half sizes of the given geoms (from their geom-frame bounding boxes)."""
    R = data.geom_xmat[geom_ids].reshape(-1, 3, 3)
    box = model.geom_aabb[geom_ids]
    center = data.geom_xpos[geom_ids] + np.einsum('nij,nj->ni', R, box[:, :3])
    half = np.einsum('nij,nj->ni', np.abs(R), box[:, 3:])
    return center, half


def _quat_xyzw_to_matrix(q: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    """Converts quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q_wxyz = _xyzw_to_wxyz(q)
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q_wxyz)
    return mat.reshape(3, 3).astype(np.float32)


class MuJoCoDigitalTwin(BaseDigitalTwin):
    """
    Physical Digital Twin of the manipulation scene running in MuJoCo.
    """

    def __init__(
        self,
        visualize: bool = False,
        table_z: float = 0.75,
        robot_model_path: Optional[str] = None,
        tracking_mode: str = "constraint",
        constraint_max_force: float = 300.0,
        kp_pos: float = 250.0,
        kd_pos: float = 30.0,
        kp_rot: float = 15.0,
        kd_rot: float = 1.5,
        sim_substeps: int = 1,
        time_step: float = 0.002,
        enable_mjx: bool = False,
        predictor: Optional[BaseObstaclePredictor] = None,
        joint_angle_offsets: Optional[List[float]] = None,
        table_in_collision_costs: bool = True,
        obstacle_convex_decomposition: bool = True,
        vhacd_resolution: int = 100000,
        vhacd_concavity: float = 0.05
    ):
        """
        :param robot_model_path: Panda MJCF (default: MuJoCo Menagerie franka_emika_panda); only its collision
            geometry is used, so the robot matches the URDF used by the PyBullet twin.
        :param obstacle_convex_decomposition: MuJoCo collides every mesh geom through its convex hull; when True,
            obstacle meshes are split into V-HACD convex parts, one geom each (same parts as the PyBullet twin).
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "[MUJOCO DIGITAL TWIN ERROR] 'mujoco' package is not installed. "
                "Please run: pip install mujoco glfw"
            )

        super().__init__(joint_angle_offsets)
        self.table_in_collision_costs = bool(table_in_collision_costs)
        self.obstacle_convex_decomposition = bool(obstacle_convex_decomposition)
        self.vhacd_resolution = int(vhacd_resolution)
        self.vhacd_concavity = float(vhacd_concavity)
        self.visualize = visualize
        self.table_z = float(table_z)
        self.enable_mjx = bool(enable_mjx)
        self.mjx_model = None
        self._eval_data = None

        # Modular generic trajectory predictor
        self.predictor: Optional[BaseObstaclePredictor] = (
            predictor if predictor is not None else ObstacleTrajectoryPredictor()
        )

        # Tracking mode: 'constraint' (or 'mocap'), 'pd_force', 'teleport'
        raw_mode = tracking_mode.lower()
        if raw_mode in ["constraint", "mocap"]:
            self.tracking_mode = "mocap"
        elif raw_mode in ["pd_force", "teleport"]:
            self.tracking_mode = raw_mode
        else:
            print(f"[MUJOCO DIGITAL TWIN WARNING] Unknown tracking mode '{tracking_mode}', defaulting to 'mocap'.")
            self.tracking_mode = "mocap"

        self.constraint_max_force = float(constraint_max_force)
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_rot = float(kp_rot)
        self.kd_rot = float(kd_rot)
        self.sim_substeps = max(1, int(sim_substeps))
        self.time_step = float(time_step)

        # Asset and Robot configuration
        if robot_model_path is not None and os.path.exists(robot_model_path):
            self.robot_model_path = os.path.abspath(robot_model_path)
        else:
            self.robot_model_path = os.path.abspath(DEFAULT_ROBOT_MODEL)
        if not os.path.exists(self.robot_model_path):
            raise FileNotFoundError(f"[MUJOCO DIGITAL TWIN ERROR] Panda MJCF not found: '{self.robot_model_path}'")
        self._robot_mjcf: Optional[Tuple[str, List[str], str]] = None
        self._model_has_robot = False
        # (geom id, body id, link index 0..7 or 8 for hand/fingers) and (geom id, obstacle id), rebuilt with the model
        self._robot_geoms: List[Tuple[int, int, int]] = []
        self._obstacle_geoms: List[Tuple[int, int]] = []

        self.robot_loaded = False
        self.robot_base_pos = np.array([0.0, 0.0, self.table_z], dtype=np.float64)
        self.robot_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # wxyz
        self.robot_id = -1  # Negative ID until load_robot() is called

        # Scene components
        self.table_spawned = False
        self.table_bounds: Optional[Tuple[float, float, float, float]] = None
        self.table_color = DEFAULT_TABLE_COLOR
        self.table_mesh_path: Optional[str] = None
        self.table_id = -1

        # Tracked dynamic objects dictionary: obj_id -> metadata
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.last_collision_timings: Dict[str, Any] = {}

        # Kinematics analytical helper
        self.kin = FrankaKinematics(base_position=self.robot_base_pos, joint_angle_offsets=self.joint_angle_offsets)

        # MuJoCo internal handles
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer = None

        # Build initial baseline model
        self._rebuild_model()

        if self.visualize:
            self._init_viewer()

        print(
            f"[MUJOCO DIGITAL TWIN] Initialized successfully "
            f"(Tracking Mode: '{self.tracking_mode}', Substeps: {self.sim_substeps}, dt: {self.time_step}s, GUI: {self.visualize})"
        )

    def _init_viewer(self) -> None:
        """Launches the interactive GLFW passive viewer if visualization is enabled."""
        if not self.visualize or self.model is None or self.data is None:
            return
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[MUJOCO DIGITAL TWIN] Passive GLFW Viewer launched.")
        except Exception as e:
            print(f"[MUJOCO DIGITAL TWIN WARNING] Could not launch passive viewer (headless environment?): {e}")
            self.viewer = None

    # -------------------------------------------------------------------------
    # In-Memory MJCF Dynamic Model Construction
    # -------------------------------------------------------------------------

    def _load_robot_mjcf(self) -> Tuple[str, List[str], str]:
        """
        Collision-only Panda from the Menagerie MJCF: the 'panda' default classes, the collision mesh assets with
        absolute paths and the link0 body tree, plus a 'tcp' site at the Panda_tip pose of the analytical model.
        """
        if self._robot_mjcf is not None:
            return self._robot_mjcf
        root = ET.parse(self.robot_model_path).getroot()
        compiler = root.find('compiler')
        mesh_dir = os.path.join(os.path.dirname(self.robot_model_path),
                                compiler.get('meshdir', '') if compiler is not None else '')
        defaults = root.find('default').find("default[@class='panda']")
        link0 = root.find('worldbody').find("body[@name='link0']")
        for body in link0.iter('body'):
            for geom in [g for g in body.findall('geom') if g.get('class') == 'visual']:
                body.remove(geom)
        used = {g.get('mesh') for g in link0.iter('geom') if g.get('mesh')}
        assets = []
        for mesh in root.find('asset').findall('mesh'):
            name = mesh.get('name') or os.path.splitext(mesh.get('file'))[0]
            if name in used:
                assets.append(f'<mesh name="{name}" file="{os.path.join(mesh_dir, mesh.get("file"))}"/>')
        link7 = link0.find(".//body[@name='link7']")
        ox, oy, oz = (float(v) for v in FrankaKinematics.EE_OFFSET)
        ET.SubElement(link7, 'site', {'name': 'tcp', 'pos': f"{ox:.6f} {oy:.6f} {0.107 + oz:.6f}"})
        self._robot_mjcf = (ET.tostring(defaults, encoding='unicode'), assets, ET.tostring(link0, encoding='unicode'))
        return self._robot_mjcf

    def _index_geoms(self) -> None:
        """Robot collision geoms by body and link, and obstacle geoms by obstacle id."""
        m = self.model
        self._robot_geoms, self._obstacle_geoms = [], []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body = int(m.geom_bodyid[g])
            name = m.body(body).name
            if name.startswith('link') and name[4:].isdigit():
                self._robot_geoms.append((g, body, int(name[4:])))
            elif name in HAND_BODIES:
                self._robot_geoms.append((g, body, 8))
            elif name.startswith('obs_'):
                try:
                    self._obstacle_geoms.append((g, int(name.split('_')[1])))
                except ValueError:
                    pass

    def _generate_mjcf_xml(self) -> str:
        """
        Dynamically constructs the complete MJCF XML string for the current scene,
        including ground plane, table, Franka Panda robot, and all discovered obstacles.
        """
        mesh_assets = []
        body_elements = []
        equality_constraints = []

        robot_defaults = ""
        if self.robot_loaded:
            robot_defaults, robot_assets, robot_body = self._load_robot_mjcf()
            mesh_assets.extend(robot_assets)
            bx, by, bz = self.robot_base_pos
            qw, qx, qy, qz = self.robot_base_quat
            body_elements.append(
                f'<body name="robot_base" pos="{bx:.6f} {by:.6f} {bz:.6f}" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">'
                f'{robot_body}</body>'
            )

        # 2. Table Structure
        if self.table_spawned:
            if self.table_bounds is not None:
                x_min, x_max, y_min, y_max = self.table_bounds
                cx = 0.5 * (x_min + x_max)
                cy = 0.5 * (y_min + y_max)
                hx = 0.5 * abs(x_max - x_min)
                hy = 0.5 * abs(y_max - y_min)
                hz = 0.5 * self.table_z
                cz = hz
                r, g, b, a = self.table_color
                table_geom = (
                    f'<geom name="table_geom" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
                    f'pos="{cx:.4f} {cy:.4f} {cz:.4f}" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" '
                    f'friction="0.5 0.005 0.0001"/>'
                )
                body_elements.append(table_geom)
            elif self.table_mesh_path is not None and os.path.exists(self.table_mesh_path):
                mesh_assets.append(f'<mesh name="table_mesh" file="{os.path.abspath(self.table_mesh_path)}"/>')
                r, g, b, a = self.table_color
                table_geom = (
                    f'<geom name="table_geom" type="mesh" mesh="table_mesh" '
                    f'rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" friction="0.5 0.005 0.0001"/>'
                )
                body_elements.append(table_geom)

        # 3. Dynamic Tracked Mesh Obstacles
        for obj_id, obj in self.tracked_objects.items():
            parts = obj.get("collision_parts") or [obj["mesh_path"]]
            for k, part in enumerate(parts):
                mesh_assets.append(f'<mesh name="mesh_obs_{obj_id}_{k}" file="{os.path.abspath(part)}"/>')
            if self.visualize and len(parts) > 1:
                mesh_assets.append(f'<mesh name="mesh_obs_{obj_id}_visual" file="{os.path.abspath(obj["mesh_path"])}"/>')

            col = obj.get("color", DEFAULT_OBSTACLE_COLOR)
            rgba_str = f"{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} {col[3]:.3f}"
            mass_val = float(obj.get("mass", 1.0)) / len(parts)
            geoms = "".join(
                f'  <geom name="obs_{obj_id}_geom_{k}" type="mesh" mesh="mesh_obs_{obj_id}_{k}" mass="{mass_val:.4f}" '
                f'rgba="{rgba_str}" friction="0.5 0.005 0.0001"/>\n' for k in range(len(parts))
            )
            if self.visualize and len(parts) > 1:
                geoms += (f'  <geom name="obs_{obj_id}_visual" type="mesh" mesh="mesh_obs_{obj_id}_visual" '
                          f'contype="0" conaffinity="0" rgba="{rgba_str}"/>\n')

            px, py, pz = obj["target_pos"]
            qw, qx, qy, qz = obj["target_quat"]

            if self.tracking_mode == "mocap":
                # Direct Kinematic Mocap Body (Perception-driven rigid obstacle)
                # In MuJoCo, placing geoms directly on a mocap body guarantees zero constraint drift,
                # instantaneous kinematic updates via eval_data/data mocap_pos, and exact collision distance queries.
                mocap_body = (
                    f'<body name="obs_{obj_id}" mocap="true" pos="{px:.6f} {py:.6f} {pz:.6f}" '
                    f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n{geoms}</body>'
                )
                body_elements.append(mocap_body)
            else:
                # Direct Dynamic Body (for PD force or teleport mode)
                dyn_body = (
                    f'<body name="obs_{obj_id}" pos="{px:.6f} {py:.6f} {pz:.6f}" '
                    f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n'
                    f'  <freejoint name="obs_{obj_id}_joint"/>\n{geoms}</body>'
                )
                body_elements.append(dyn_body)

        assets_block = "\n    ".join(mesh_assets)
        world_block = "\n    ".join(body_elements)
        eq_block = "\n    ".join(equality_constraints)

        xml = f"""<mujoco model="drema_digital_twin">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{self.time_step}" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom contype="1" conaffinity="1" density="1000"/>
    {robot_defaults}
  </default>

  <asset>
    {assets_block}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1" friction="1 0.005 0.0001"/>
    {world_block}
  </worldbody>

  <equality>
    {eq_block}
  </equality>
</mujoco>
"""
        return xml

    def _rebuild_model(self) -> None:
        """
        Recompiles the MuJoCo C model from the dynamically updated MJCF XML,
        preserving previous robot joint states and dynamic obstacle states across rebuilds.
        """
        # Save previous states if existing
        # Only a previous model that already contained the robot holds robot joint values in qpos[:7]; otherwise
        # those entries belong to an obstacle free joint.
        prev_robot_q = None
        if self.model is not None and self.data is not None and self._model_has_robot:
            prev_robot_q = self.data.qpos[:7].copy()

        xml = self._generate_mjcf_xml()
        new_model = mujoco.MjModel.from_xml_string(xml)
        new_data = mujoco.MjData(new_model)

        # Restore states
        if self.robot_loaded:
            if prev_robot_q is not None:
                new_data.qpos[:7] = prev_robot_q
            for finger in ("finger_joint1", "finger_joint2"):
                jid = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_JOINT, finger)
                if jid >= 0:
                    new_data.qpos[new_model.jnt_qposadr[jid]] = FINGER_OPENING

        # Initialize mocap positions for obstacles
        if self.tracking_mode == "mocap":
            for obj_id, obj in self.tracked_objects.items():
                body_name = f"obs_{obj_id}"
                body_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    body_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}_mocap")
                if body_id >= 0:
                    mocap_idx = new_model.body_mocapid[body_id]
                    if mocap_idx >= 0:
                        new_data.mocap_pos[mocap_idx] = obj["target_pos"]
                        new_data.mocap_quat[mocap_idx] = obj["target_quat"]

        mujoco.mj_forward(new_model, new_data)
        self.model = new_model
        self.data = new_data
        self._eval_data = mujoco.MjData(new_model)
        self._model_has_robot = self.robot_loaded
        self._index_geoms()

        if self.enable_mjx:
            try:
                from mujoco import mjx
                self.mjx_model = mjx.put_model(new_model)
                print("[MUJOCO DIGITAL TWIN] ✓ Synchronized model to MJX device memory.")
            except Exception as e:
                print(f"[MUJOCO DIGITAL TWIN WARNING] MJX initialization deferred: {e}")
                self.mjx_model = None

    # -------------------------------------------------------------------------
    # Robot Interface Implementation (BaseDigitalTwin contract)
    # -------------------------------------------------------------------------

    @synchronized
    def load_robot(
        self,
        base_position: Tuple[float, float, float],
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        joint_positions: Optional[List[float]] = None
    ) -> bool:
        """Loads or updates the manipulator arm at the specified world base pose."""
        self.robot_base_pos = np.array(base_position, dtype=np.float64)
        self.robot_base_quat = _xyzw_to_wxyz(base_orientation)
        self.robot_loaded = True
        self.robot_id = 1
        self.kin = FrankaKinematics(base_position=self.robot_base_pos, joint_angle_offsets=self.joint_angle_offsets)

        self._rebuild_model()

        if joint_positions is not None:
            self.sync_robot_state(joint_positions)

        print(
            f"[MUJOCO DIGITAL TWIN] Loaded Franka Panda robot at "
            f"[{self.robot_base_pos[0]:.3f}, {self.robot_base_pos[1]:.3f}, {self.robot_base_pos[2]:.3f}]"
        )
        return True

    @synchronized
    def sync_robot_state(
        self,
        joint_positions: List[float],
        joint_velocities: Optional[List[float]] = None
    ) -> None:
        """Synchronizes the physical twin robot joints with current kinematic state."""
        if not self.robot_loaded or self.data is None:
            return

        num_j = min(len(joint_positions), 7, self.model.nq)
        self.data.qpos[:num_j] = np.array(joint_positions[:num_j], dtype=np.float64) + self.joint_angle_offsets[:num_j]
        if joint_velocities is not None:
            self.data.qvel[:num_j] = np.array(joint_velocities[:num_j], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

    @synchronized
    def get_joint_positions(self) -> List[float]:
        """Returns the current 7-DoF robot arm joint angles [rad]."""
        if not self.robot_loaded or self.data is None:
            return [0.0] * 7
        return [float(x) for x in self.data.qpos[:7] - self.joint_angle_offsets]

    @synchronized
    def get_joint_velocities(self) -> List[float]:
        """Returns the current 7-DoF robot arm joint velocities [rad/s]."""
        if not self.robot_loaded or self.data is None:
            return [0.0] * 7
        return [float(x) for x in self.data.qvel[:7]]

    @synchronized
    def get_ee_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Returns the current Cartesian pose of the end-effector (TCP site) in world coordinates."""
        if not self.robot_loaded or self.model is None or self.data is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

        tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if tcp_site_id >= 0:
            pos = tuple(float(x) for x in self.data.site_xpos[tcp_site_id])
            rot_mat = self.data.site_xmat[tcp_site_id].reshape(3, 3)
            quat = _matrix_to_quat_xyzw(rot_mat)
            return (pos, quat)

        # Fallback to analytical forward kinematics
        q = np.array(self.get_joint_positions(), dtype=np.float32)
        pos, rot_mat = self.kin.forward_kinematics_ee(q)
        quat = _matrix_to_quat_xyzw(rot_mat)
        return (tuple(float(x) for x in pos), quat)

    @synchronized
    def compute_inverse_kinematics(
        self,
        target_position: Tuple[float, float, float],
        target_orientation: Optional[Tuple[float, float, float, float]] = None
    ) -> List[float]:
        """Calculates 7-DoF joint angles to reach the given target end-effector pose."""
        q_init = np.array(self.get_joint_positions(), dtype=np.float32)
        rot_mat = None
        if target_orientation is not None:
            rot_mat = _quat_xyzw_to_matrix(target_orientation)

        q_sol, success = self.kin.solve_dls_ik(q_init, np.array(target_position, dtype=np.float32), rot_mat)
        if success:
            return [float(x) for x in q_sol]
        return [float(x) for x in q_init]

    # -------------------------------------------------------------------------
    # Environment & Object Spawning (BaseDigitalTwin contract)
    # -------------------------------------------------------------------------

    @synchronized
    def spawn_scanned_table(
        self,
        table_z: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        mesh_file_path: Optional[str] = None,
        color: Tuple[float, float, float, float] = DEFAULT_TABLE_COLOR
    ) -> int:
        """Spawns the tabletop support structure discovered by initial 3D scan."""
        self.table_z = float(table_z)
        self.table_bounds = bounds
        self.table_mesh_path = mesh_file_path
        self.table_color = color
        self.table_spawned = True
        self.table_id = 2  # Standard ID for interface consistency

        self._rebuild_model()
        print(f"[MUJOCO DIGITAL TWIN] Spawned scanned table structure (Z={self.table_z:.3f}m)")
        return self.table_id

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
        """Spawns an extracted Marching Cubes surface mesh obstacle into MuJoCo physics."""
        target_obj_id = obj_id if obj_id is not None else len(self.tracked_objects)

        col = tuple(color) if color is not None else DEFAULT_OBSTACLE_COLOR
        quat_wxyz = _xyzw_to_wxyz(initial_quat)

        parts = [mesh_path]
        if self.obstacle_convex_decomposition:
            try:
                parts = convex_parts(mesh_path, self.vhacd_resolution, self.vhacd_concavity)
            except Exception as e:
                print(f"[MUJOCO DIGITAL TWIN WARNING] Convex decomposition failed ({e}): using the convex hull of '{mesh_path}'")

        self.tracked_objects[target_obj_id] = {
            "name": name,
            "mesh_path": mesh_path,
            "collision_parts": parts,
            "mass": float(mass),
            "color": col,
            "target_pos": np.array(initial_pos, dtype=np.float64),
            "target_quat": quat_wxyz,
            "body_id": target_obj_id + 10  # Virtual body ID for caller queries
        }

        self._rebuild_model()
        print(
            f"[MUJOCO DIGITAL TWIN] Spawned mesh obstacle ID {target_obj_id} "
            f"('{name}', Mass={mass}kg, Mode={self.tracking_mode})"
        )
        return target_obj_id

    @synchronized
    def sync_object_pose(
        self,
        obj_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        timestamp: Optional[float] = None
    ) -> None:
        """Updates rigid body position and orientation estimated by SE(3) tracking."""
        if obj_id not in self.tracked_objects or self.data is None:
            return

        pos_arr = np.array(position, dtype=np.float64)
        quat_wxyz = _xyzw_to_wxyz(orientation)

        self.tracked_objects[obj_id]["target_pos"] = pos_arr
        self.tracked_objects[obj_id]["target_quat"] = quat_wxyz

        # Update generic trajectory predictor
        if self.predictor is not None:
            self.predictor.update_obstacle_pose(obj_id, position, orientation, timestamp=timestamp)

        if self.tracking_mode == "mocap":
            body_name = f"obs_{obj_id}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}_mocap")
            if body_id >= 0:
                mocap_idx = self.model.body_mocapid[body_id]
                if mocap_idx >= 0:
                    self.data.mocap_pos[mocap_idx] = pos_arr
                    self.data.mocap_quat[mocap_idx] = quat_wxyz
                    mujoco.mj_forward(self.model, self.data)
        elif self.tracking_mode == "teleport":
            body_name = f"obs_{obj_id}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                jnt_id = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr:qpos_adr+3] = pos_arr
                self.data.qpos[qpos_adr+3:qpos_adr+7] = quat_wxyz
                # Zero out velocities on kinematic reset
                dof_adr = self.model.jnt_dofadr[jnt_id]
                self.data.qvel[dof_adr:dof_adr+6] = 0.0

    @synchronized
    def get_object_pose(
        self,
        obj_id: int
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Retrieves physical rigid body position and orientation for the given object ID."""
        if obj_id not in self.tracked_objects or self.model is None or self.data is None:
            return None

        body_name = f"obs_{obj_id}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}_mocap")
        if body_id < 0:
            return None

        pos = tuple(float(x) for x in self.data.xpos[body_id])
        quat = _wxyz_to_xyzw(self.data.xquat[body_id])
        return (pos, quat)

    @synchronized
    def remove_object(self, obj_id: int) -> bool:
        """Removes the specified dynamic mesh object from the simulation."""
        if obj_id in self.tracked_objects:
            del self.tracked_objects[obj_id]
            self._rebuild_model()
            return True
        return False

    # -------------------------------------------------------------------------
    # Physics Stepping & Collision Checking
    # -------------------------------------------------------------------------

    @synchronized
    def step_simulation(self) -> None:
        """Advances physical simulation step and synchronizes visualizer."""
        if self.model is None or self.data is None:
            return

        for _ in range(self.sim_substeps):
            # Apply PD forces if operating in pd_force mode
            if self.tracking_mode == "pd_force":
                self._apply_pd_tracking_forces()

            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def _apply_pd_tracking_forces(self) -> None:
        """Calculates and applies 6D spatial PD tracking forces and torques + gravity feedforward."""
        for obj_id, obj in self.tracked_objects.items():
            body_name = f"obs_{obj_id}"
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if b_id < 0:
                continue

            # Linear tracking
            p_curr = self.data.xpos[b_id]
            p_targ = obj["target_pos"]
            jnt_id = self.model.body_jntadr[b_id]
            dof_adr = self.model.jnt_dofadr[jnt_id]
            v_curr = self.data.qvel[dof_adr:dof_adr+3]

            mass_val = obj.get("mass", 1.0)
            f_pd = self.kp_pos * (p_targ - p_curr) - self.kd_pos * v_curr
            f_grav = np.array([0.0, 0.0, mass_val * 9.81], dtype=np.float64)
            f_total = f_pd + f_grav

            # Angular tracking
            q_curr = self.data.xquat[b_id] # wxyz
            q_targ = obj["target_quat"]    # wxyz
            w_curr = self.data.qvel[dof_adr+3:dof_adr+6]

            # Orientation error in Lie algebra so(3)
            # q_err = q_targ * conj(q_curr)
            w1, x1, y1, z1 = q_targ
            w2, x2, y2, z2 = q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]
            w_err = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            v_err = np.array([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            ], dtype=np.float64)
            if w_err < 0:
                v_err = -v_err
            tau_pd = 2.0 * self.kp_rot * v_err - self.kd_rot * w_curr

            # MuJoCo xfrc_applied format: [torque_x, torque_y, torque_z, force_x, force_y, force_z]
            self.data.xfrc_applied[b_id, :3] = f_total
            self.data.xfrc_applied[b_id, 3:6] = tau_pd

    @synchronized
    def check_collision(self) -> bool:
        """Returns True if there is contact between the robot and any scene obstacle or table."""
        if self.model is None or self.data is None:
            return False

        robot_body_ids = {body for _, body, _ in self._robot_geoms}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            if (b1 in robot_body_ids and b2 not in robot_body_ids) or (b2 in robot_body_ids and b1 not in robot_body_ids):
                # Ignore contacts with the floor
                if "floor" in self.model.geom(contact.geom1).name or "floor" in self.model.geom(contact.geom2).name:
                    continue
                return True
        return False

    @synchronized
    def get_min_obstacle_distance(self) -> float:
        """Computes the signed minimum Euclidean distance between robot arm links and dynamic obstacles."""
        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return float("inf")

        robot_geoms = [g for g, _, _ in self._robot_geoms]
        obstacle_geoms = [g for g, _ in self._obstacle_geoms]

        if not robot_geoms or not obstacle_geoms:
            return float("inf")

        min_d = float("inf")
        fromto = np.zeros(6, dtype=np.float64)
        for rg in robot_geoms:
            for og in obstacle_geoms:
                try:
                    d = mujoco.mj_geomDistance(self.model, self.data, rg, og, 1.0, fromto)
                    if d < min_d:
                        min_d = d
                except Exception:
                    pass
        return float(min_d)

    @synchronized
    def get_closest_points(self, distance: float = 0.5) -> List[Tuple[int, int, float, Tuple[float, float, float]]]:
        """
        Calculates closest points between robot and obstacles within specified distance limit.
        Returns list of (robot_geom_id, obstacle_geom_id, distance, contact_point_on_robot).
        """
        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return []

        robot_geoms = [g for g, _, link in self._robot_geoms if link >= 1]
        obstacle_geoms = [g for g, _ in self._obstacle_geoms]

        results = []
        fromto = np.zeros(6, dtype=np.float64)
        for rg in robot_geoms:
            for og in obstacle_geoms:
                try:
                    d = mujoco.mj_geomDistance(self.model, self.data, rg, og, distance, fromto)
                    if d <= distance:
                        pt = (float(fromto[0]), float(fromto[1]), float(fromto[2]))
                        results.append((rg, og, float(d), pt))
                except Exception:
                    pass
        return results

    # -------------------------------------------------------------------------
    # MPC & Parallel GPU Acceleration Helpers (MJX / Warp bridge)
    # -------------------------------------------------------------------------

    def get_mj_model(self) -> Optional[mujoco.MjModel]:
        """Provides direct access to the compiled MjModel C struct for parallel GPU rollout bridges."""
        return self.model

    def get_mj_data(self) -> Optional[mujoco.MjData]:
        """Provides direct access to current MjData simulation state for cloning."""
        return self.data

    @synchronized
    def get_tracked_obstacles_info(self) -> List[Dict[str, Any]]:
        """Retrieves list of tracked dynamic obstacle dictionaries."""
        obstacles = []
        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return obstacles

        for obj_id, obj_data in self.tracked_objects.items():
            pose = self.get_object_pose(obj_id)
            if pose is not None:
                pos, orn = pose
            else:
                pos = tuple(float(x) for x in obj_data.get('target_pos', [0, 0, 0]))
                quat_wxyz = obj_data.get('target_quat', [1, 0, 0, 0])
                orn = _wxyz_to_xyzw(quat_wxyz)

            obstacles.append({
                'id': obj_id,
                'name': obj_data.get('name', f"obstacle_{obj_id}"),
                'position': pos,
                'orientation': orn,
                'velocity': self._estimated_velocity(obj_id),
                'is_target': bool(obj_data.get('is_target', False))
            })
        return obstacles

    def _estimated_velocity(self, obj_id: int) -> Tuple[float, float, float]:
        if self.predictor is not None:
            st = self.predictor.get_estimated_state(obj_id)
            if st is not None:
                return tuple(float(x) for x in st['velocity'])
        return (0.0, 0.0, 0.0)

    @synchronized
    def get_obstacle_proximity(self, max_distance: float) -> List[Dict[str, Any]]:
        """Closest robot/obstacle surface points per (obstacle, link) pair, via mj_geomDistance."""
        if self.model is None or self.data is None or not self.robot_loaded:
            return []

        # Hand and fingers are rigidly attached to link 7, as in the PyBullet twin.
        robot_geoms = [(g, min(link, 7)) for g, _, link in self._robot_geoms if link >= 1]

        closest: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]] = {}
        fromto = np.zeros(6, dtype=np.float64)
        for og, obj_id in self._obstacle_geoms:
            if self.tracked_objects.get(obj_id, {}).get("is_target", False):
                continue
            for rg, link_idx in robot_geoms:
                d = mujoco.mj_geomDistance(self.model, self.data, rg, og, float(max_distance), fromto)
                if d >= max_distance:
                    continue
                diff = fromto[0:3] - fromto[3:6]
                norm = np.linalg.norm(diff)
                if norm < 1e-9:
                    continue
                # fromto runs from the robot geom to the obstacle geom; when penetrating, the
                # nearest-point pair crosses, so the direction must be flipped.
                normal = diff / norm if d >= 0.0 else -diff / norm
                key = (obj_id, link_idx)
                if key not in closest or d < closest[key][0]:
                    closest[key] = (float(d), fromto[0:3].copy(), normal)

        return [
            {
                'obj_id': obj_id,
                'link_idx': link_idx,
                'point_world': point,
                'normal': normal,
                'distance': d,
                'velocity': self._estimated_velocity(obj_id)
            }
            for (obj_id, link_idx), (d, point, normal) in closest.items()
        ]

    @synchronized
    def calculate_inverse_kinematics(
        self,
        target_pos: Tuple[float, float, float],
        target_quat: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """Calculates inverse kinematics solution for Franka Panda end-effector using MuJoCo Jacobian."""
        if self.model is None or self.data is None or self.model.nq < 7:
            return None

        if self._eval_data is None:
            self._eval_data = mujoco.MjData(self.model)

        eval_data = self._eval_data
        if np.all(np.abs(self.data.qpos[:7]) < 1e-2):
            eval_data.qpos[:7] = FrankaKinematics.Q_REST.astype(np.float64) + self.joint_angle_offsets
        else:
            eval_data.qpos[:7] = self.data.qpos[:7]

        # Parse target orientation if provided (supports 3x3 matrix or 4-element quaternion)
        rot_mat = None
        if target_quat is not None:
            tq = np.asarray(target_quat)
            if tq.shape == (3, 3):
                rot_mat = tq.astype(np.float64)
            elif len(tq.flatten()) == 4:
                rot_mat = Rotation.from_quat(tq.flatten()).as_matrix().astype(np.float64)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if site_id < 0 and body_id < 0:
            return None

        target_p = np.array(target_pos, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64) if rot_mat is not None else None

        q_min = FrankaKinematics.Q_MIN.astype(np.float64) + self.joint_angle_offsets
        q_max = FrankaKinematics.Q_MAX.astype(np.float64) + self.joint_angle_offsets

        for _ in range(40):
            mujoco.mj_fwdPosition(self.model, eval_data)
            curr_pos = eval_data.site_xpos[site_id] if site_id >= 0 else eval_data.xpos[body_id]
            pos_err = target_p - curr_pos

            if rot_mat is not None:
                curr_mat = eval_data.site_xmat[site_id].reshape(3, 3) if site_id >= 0 else eval_data.xmat[body_id].reshape(3, 3)
                rot_err = 0.5 * (
                    np.cross(curr_mat[:, 0], rot_mat[:, 0]) +
                    np.cross(curr_mat[:, 1], rot_mat[:, 1]) +
                    np.cross(curr_mat[:, 2], rot_mat[:, 2])
                )
                if np.linalg.norm(pos_err) < 2e-3 and np.linalg.norm(rot_err) < 0.05:
                    return (eval_data.qpos[:7] - self.joint_angle_offsets).astype(np.float32)

                if site_id >= 0:
                    mujoco.mj_jacSite(self.model, eval_data, jacp, jacr, site_id)
                else:
                    mujoco.mj_jacBody(self.model, eval_data, jacp, jacr, body_id)

                J = np.vstack([jacp[:, :7], jacr[:, :7]])
                err = np.concatenate([pos_err, 0.2 * rot_err])
                step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            else:
                if np.linalg.norm(pos_err) < 2e-3:
                    return (eval_data.qpos[:7] - self.joint_angle_offsets).astype(np.float32)

                if site_id >= 0:
                    mujoco.mj_jacSite(self.model, eval_data, jacp, None, site_id)
                else:
                    mujoco.mj_jacBody(self.model, eval_data, jacp, None, body_id)

                J = jacp[:, :7]
                step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), pos_err)

            eval_data.qpos[:7] = np.clip(eval_data.qpos[:7] + 0.6 * step, q_min, q_max)

        mujoco.mj_fwdPosition(self.model, eval_data)
        curr_pos = eval_data.site_xpos[site_id] if site_id >= 0 else eval_data.xpos[body_id]
        if np.linalg.norm(target_p - curr_pos) < 0.05:
            return (eval_data.qpos[:7] - self.joint_angle_offsets).astype(np.float32)

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
        a batch of candidate trajectories Q (K x H x 7) and QD (K x H x 7) using MuJoCo geomDistance / MJX.
        Obstacle future states are dynamically forecasted across horizon H via self.predictor.
        """
        K, H, _ = Q.shape
        coll_p = np.zeros((K, H), dtype=np.float32)
        coll_gvm = np.zeros((K, H), dtype=np.float32)

        if self.model is None or self.data is None:
            self.last_collision_timings = {
                'backend': 'MJX' if getattr(self, 'enable_mjx', False) else 'MuJoCo',
                'pred_ms': 0.0,
                'backend_ms': 0.0,
                'gvm_ms': 0.0,
                'pts_count': 0
            }
            return coll_p, coll_gvm

        # Obstacle geoms (several convex parts per obstacle) and the obstacle slot used for the cost grouping:
        # potentials are aggregated per (obstacle, robot body), as in the PyBullet twin.
        obstacle_geoms = []
        geom_to_obj_id = {}
        geom_slot = {}
        slots: Dict[int, int] = {}
        for g, obj_id in self._obstacle_geoms:
            if self.tracked_objects.get(obj_id, {}).get("is_target", False):
                continue
            obstacle_geoms.append(g)
            geom_to_obj_id[g] = obj_id
            geom_slot[g] = slots.setdefault(obj_id, len(slots))
        # The table is static (no prediction, zero velocity); the fixed base link0 is left out of the robot
        # geoms below, so only the moving links are penalized against it.
        if self.table_in_collision_costs:
            table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
            if table_geom_id >= 0:
                obstacle_geoms.append(table_geom_id)
                geom_slot[table_geom_id] = len(slots)
                slots[-1] = len(slots)

        if not obstacle_geoms:
            self.last_collision_timings = {
                'backend': 'MJX' if getattr(self, 'enable_mjx', False) else 'MuJoCo',
                'pred_ms': 0.0,
                'backend_ms': 0.0,
                'gvm_ms': 0.0,
                'pts_count': 0
            }
            return coll_p, coll_gvm

        # Moving robot geoms: (geom, body, velocity slot of batch_link_linear_velocities; hand/fingers -> TCP)
        robot_geom_link_map = [(g, body, link - 1 if link <= 7 else 7) for g, body, link in self._robot_geoms if link >= 1]
        robot_geom_ids = np.array([g for g, _, _ in robot_geom_link_map], dtype=np.int64)
        obstacle_geom_ids = np.array(obstacle_geoms, dtype=np.int64)
        n_bodies = self.model.nbody

        if not robot_geom_link_map:
            self.last_collision_timings = {
                'backend': 'MJX' if getattr(self, 'enable_mjx', False) else 'MuJoCo',
                'pred_ms': 0.0,
                'backend_ms': 0.0,
                'gvm_ms': 0.0,
                'pts_count': 0
            }
            return coll_p, coll_gvm

        if self._eval_data is None:
            self._eval_data = mujoco.MjData(self.model)

        eval_data = self._eval_data
        eval_data.qpos[:] = self.data.qpos[:]
        if hasattr(eval_data, 'mocap_pos') and hasattr(self.data, 'mocap_pos'):
            eval_data.mocap_pos[:] = self.data.mocap_pos[:]
            eval_data.mocap_quat[:] = self.data.mocap_quat[:]

        # Forecast future trajectories for all tracked dynamic obstacles across horizon H
        t_pred_start = time.perf_counter()
        predictions = (
            self.predictor.predict_all(horizon=H, dt=dt)
            if self.predictor is not None
            else {}
        )
        t_pred_ms = (time.perf_counter() - t_pred_start) * 1000.0

        t_mj_total = 0.0
        t_gvm_total = 0.0
        total_contact_pts = 0

        # Pre-cache mocap indices for tracked obstacles
        obj_mocap_map = {}
        for obj_id in predictions.keys():
            m_name = f"obs_{obj_id}"
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, m_name)
            if b_id < 0:
                b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}_mocap")
            if b_id >= 0:
                m_idx = self.model.body_mocapid[b_id]
                if m_idx >= 0:
                    obj_mocap_map[obj_id] = m_idx

        # Subsample lookahead waypoints to sustain high control frequency
        step_stride = max(1, H // 5)
        eval_steps = list(range(0, H, step_stride))
        if (H - 1) not in eval_steps:
            eval_steps.append(H - 1)

        fromto = np.zeros(6, dtype=np.float64)

        # Closest-point records, aggregated per (obstacle, robot body).
        rec_k: List[int] = []
        rec_e: List[int] = []
        rec_group: List[int] = []
        rec_slot: List[int] = []
        rec_d: List[float] = []
        rec_normal: List[Any] = []
        rec_v_obs: List[Any] = []
        rec_point: List[Any] = []
        rec_obj: List[int] = []
        record_points = self.record_collision_points
        zero_vel = np.zeros(3, dtype=np.float64)

        for e, h in enumerate(eval_steps):
            # Advance all tracked dynamic obstacles to forecasted pose at step h
            for obj_id, m_idx in obj_mocap_map.items():
                if obj_id in predictions:
                    pred = predictions[obj_id]
                    eval_data.mocap_pos[m_idx] = pred.positions[h].astype(np.float64)
                    eval_data.mocap_quat[m_idx] = _xyzw_to_wxyz(pred.orientations[h])
            obs_vel = {}
            for og in obstacle_geoms:
                o_id = geom_to_obj_id.get(og)
                obs_vel[og] = predictions[o_id].velocities[h] if o_id is not None and o_id in predictions else zero_vel

            for k in range(K):
                q_step = Q[k, h]

                # Update Franka Panda joint positions in MuJoCo eval data
                t_mj_s = time.perf_counter()
                eval_data.qpos[:7] = q_step + self.joint_angle_offsets
                mujoco.mj_kinematics(self.model, eval_data)
                t_mj_total += (time.perf_counter() - t_mj_s)

                # Broadphase: only geom pairs whose world AABBs, grown by sigma_2, overlap reach the GJK query.
                r_center, r_half = _world_aabbs(self.model, eval_data, robot_geom_ids)
                o_center, o_half = _world_aabbs(self.model, eval_data, obstacle_geom_ids)
                near = np.all(np.abs(r_center[:, None] - o_center[None]) <= r_half[:, None] + o_half[None] + sigma_2,
                              axis=-1)
                for ri, oi in zip(*np.nonzero(near)):
                    rg, rbody, vel_slot = robot_geom_link_map[ri]
                    og = obstacle_geoms[oi]
                    t_mj_s = time.perf_counter()
                    try:
                        d = mujoco.mj_geomDistance(self.model, eval_data, rg, og, float(sigma_2), fromto)
                    except Exception:
                        t_mj_total += (time.perf_counter() - t_mj_s)
                        continue
                    t_mj_total += (time.perf_counter() - t_mj_s)

                    # mj_geomDistance returns distmax itself when the pair is farther than distmax
                    if d >= sigma_2:
                        continue

                    total_contact_pts += 1
                    rec_k.append(k)
                    rec_e.append(e)
                    rec_group.append(geom_slot[og] * n_bodies + rbody)
                    rec_slot.append(vel_slot)
                    rec_d.append(d)
                    # obstacle -> robot direction; the nearest-point pair crosses when penetrating
                    rec_normal.append(fromto[0:3] - fromto[3:6] if d >= 0.0 else fromto[3:6] - fromto[0:3])
                    rec_v_obs.append(obs_vel[og])
                    if record_points:
                        rec_point.append(fromto[0:3].copy())
                        rec_obj.append(geom_to_obj_id.get(og, -1))

        if record_points:
            self.last_collision_points = {
                'rollout': np.asarray(rec_k, dtype=np.int64),
                'step': np.asarray(eval_steps, dtype=np.int64)[np.asarray(rec_e, dtype=np.int64)],
                'position': np.asarray(rec_point, dtype=np.float64).reshape(-1, 3),
                'distance': np.asarray(rec_d, dtype=np.float64),
                'obstacle': np.asarray(rec_obj, dtype=np.int64),
            }

        t_gvm_s = time.perf_counter()
        coll_p_eval, coll_gvm_eval = aggregate_gvm_costs(
            Q[:, eval_steps], QD[:, eval_steps], rec_k, rec_e, rec_group, rec_slot, rec_d, rec_normal, rec_v_obs,
            n_groups=max(1, len(slots)) * n_bodies, sigma_1=sigma_1, sigma_2=sigma_2, kappa=kappa, rho=rho,
            kin_helper=kin_helper
        )
        coll_p[:, eval_steps] = coll_p_eval
        coll_gvm[:, eval_steps] = coll_gvm_eval
        t_gvm_total = time.perf_counter() - t_gvm_s

        self.last_collision_timings = {
            'backend': 'MJX' if getattr(self, 'enable_mjx', False) else 'MuJoCo',
            'pred_ms': t_pred_ms,
            'backend_ms': t_mj_total * 1000.0,
            'gvm_ms': t_gvm_total * 1000.0,
            'pts_count': total_contact_pts
        }

        # Restore eval_data mocap poses to instantaneous state (t=0)
        if hasattr(eval_data, 'mocap_pos') and hasattr(self.data, 'mocap_pos'):
            eval_data.mocap_pos[:] = self.data.mocap_pos[:]
            eval_data.mocap_quat[:] = self.data.mocap_quat[:]

        # Interpolate across un-evaluated lookahead steps
        for h in range(H):
            if h not in eval_steps:
                prev_h = max([s for s in eval_steps if s <= h], default=0)
                coll_p[:, h] = coll_p[:, prev_h]
                coll_gvm[:, h] = coll_gvm[:, prev_h]

        return coll_p, coll_gvm

    # -------------------------------------------------------------------------
    # Lifecycle Cleanup
    # -------------------------------------------------------------------------

    @synchronized
    def reset(self) -> None:
        """Resets the simulation environment and cleans up dynamic bodies."""
        self.tracked_objects.clear()
        if self.predictor is not None:
            self.predictor.reset()
        self._rebuild_model()
        print("[MUJOCO DIGITAL TWIN] Environment reset.")

    @synchronized
    def shutdown(self) -> None:
        """Cleans up physics server and closes visualizer."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self.model = None
        self.data = None
        self._eval_data = None
        self.mjx_model = None
        print("[MUJOCO DIGITAL TWIN] Physics server shutdown.")

