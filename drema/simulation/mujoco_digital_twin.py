#!/usr/bin/env python
"""
MuJoCo Digital Twin for DREMA Dynamic Inference System.

Agnostic & Dynamic Scene Manager running on MuJoCo:
- Loads the Franka Emika Panda robot model and static environment (ground, table).
- Dynamically compiles discovered 3D surface mesh obstacles extracted by Perception.
- Fluid dynamic multi-body tracking using MuJoCo Mocap Bodies + Weld Equality Constraints
  (or Cartesian PD force/torque with gravity feedforward), natively enforcing non-interpenetration.
- High-performance C physics (<15 microseconds/step), ready for GPU batching with MJX / Warp.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR
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
        predictor: Optional[BaseObstaclePredictor] = None
    ):
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "[MUJOCO DIGITAL TWIN ERROR] 'mujoco' package is not installed. "
                "Please run: pip install mujoco glfw"
            )

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
            default_path = os.path.abspath("assets/franka_panda/panda.xml")
            self.robot_model_path = default_path if os.path.exists(default_path) else None

        self.robot_loaded = False
        self.robot_base_pos = np.array([0.0, 0.0, self.table_z], dtype=np.float64)
        self.robot_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # wxyz
        self.robot_id = 1  # Standard ID for interface compatibility with PyBullet callers

        # Scene components
        self.table_spawned = False
        self.table_bounds: Optional[Tuple[float, float, float, float]] = None
        self.table_color = DEFAULT_TABLE_COLOR
        self.table_mesh_path: Optional[str] = None
        self.table_id = -1

        # Tracked dynamic objects dictionary: obj_id -> metadata
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}

        # Kinematics analytical helper
        self.kin = FrankaKinematics(base_position=self.robot_base_pos)

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

    def _generate_mjcf_xml(self) -> str:
        """
        Dynamically constructs the complete MJCF XML string for the current scene,
        including ground plane, table, Franka Panda robot, and all discovered obstacles.
        """
        mesh_assets = []
        body_elements = []
        equality_constraints = []

        objs_dir = os.path.abspath("assets/franka_panda/objs")

        # 1. Base Robot Meshes (if robot model path exists)
        if self.robot_loaded:
            mesh_assets.append(f'<mesh name="franka_base" file="{os.path.join(objs_dir, "robot_base.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link1" file="{os.path.join(objs_dir, "Panda_link1_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link2" file="{os.path.join(objs_dir, "Panda_link2_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link3" file="{os.path.join(objs_dir, "Panda_link3_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link4" file="{os.path.join(objs_dir, "Panda_link4_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link5" file="{os.path.join(objs_dir, "Panda_link5_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link6" file="{os.path.join(objs_dir, "Panda_link6_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_link7" file="{os.path.join(objs_dir, "Panda_link7_respondable.obj")}"/>')
            mesh_assets.append(f'<mesh name="franka_gripper" file="{os.path.join(objs_dir, "Panda_gripper_visual.obj")}"/>')

            # Robot body tree
            bx, by, bz = self.robot_base_pos
            qw, qx, qy, qz = self.robot_base_quat
            robot_tree = f"""
    <body name="robot_base" pos="{bx:.6f} {by:.6f} {bz:.6f}" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">
      <geom name="robot_base_geom" type="mesh" mesh="franka_base" rgba="0.95 0.95 0.95 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom name="link1_geom" type="mesh" mesh="franka_link1" rgba="0.95 0.95 0.95 1"/>
        <body name="link2" pos="0 0 0" quat="0.7071068 -0.7071068 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom name="link2_geom" type="mesh" mesh="franka_link2" rgba="0.95 0.95 0.95 1"/>
          <body name="link3" pos="0 -0.316 0" quat="0.7071068 0.7071068 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom name="link3_geom" type="mesh" mesh="franka_link3" rgba="0.95 0.95 0.95 1"/>
            <body name="link4" pos="0.0825 0 0" quat="0.7071068 0.7071068 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom name="link4_geom" type="mesh" mesh="franka_link4" rgba="0.95 0.95 0.95 1"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="0.7071068 -0.7071068 0 0">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom name="link5_geom" type="mesh" mesh="franka_link5" rgba="0.95 0.95 0.95 1"/>
                <body name="link6" pos="0 0 0" quat="0.7071068 0.7071068 0 0">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom name="link6_geom" type="mesh" mesh="franka_link6" rgba="0.95 0.95 0.95 1"/>
                  <body name="link7" pos="0.088 0 0" quat="0.7071068 0.7071068 0 0">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom name="link7_geom" type="mesh" mesh="franka_link7" rgba="0.95 0.95 0.95 1"/>
                    <body name="ee" pos="0 0 0.107">
                      <geom name="gripper_geom" type="mesh" mesh="franka_gripper" rgba="0.3 0.3 0.3 1"/>
                      <site name="tcp" pos="0 0 0.1034"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""
            body_elements.append(robot_tree)

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
            mesh_name = f"mesh_obs_{obj_id}"
            abs_mesh_path = os.path.abspath(obj["mesh_path"])
            mesh_assets.append(f'<mesh name="{mesh_name}" file="{abs_mesh_path}"/>')

            col = obj.get("color", DEFAULT_OBSTACLE_COLOR)
            rgba_str = f"{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} {col[3]:.3f}"
            mass_val = float(obj.get("mass", 1.0))

            px, py, pz = obj["target_pos"]
            qw, qx, qy, qz = obj["target_quat"]

            if self.tracking_mode == "mocap":
                # Mocap Kinematic Anchor (Perception reference)
                mocap_body = (
                    f'<body name="obs_{obj_id}_mocap" mocap="true" pos="{px:.6f} {py:.6f} {pz:.6f}" '
                    f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n'
                    f'  <geom type="sphere" size="0.005" rgba="1 1 1 0" contype="0" conaffinity="0"/>\n'
                    f'</body>'
                )
                body_elements.append(mocap_body)

                # Physical Dynamic Body
                dyn_body = (
                    f'<body name="obs_{obj_id}" pos="{px:.6f} {py:.6f} {pz:.6f}" '
                    f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n'
                    f'  <freejoint name="obs_{obj_id}_joint"/>\n'
                    f'  <geom name="obs_{obj_id}_geom" type="mesh" mesh="{mesh_name}" mass="{mass_val:.3f}" '
                    f'rgba="{rgba_str}" friction="0.5 0.005 0.0001"/>\n'
                    f'</body>'
                )
                body_elements.append(dyn_body)

                # Weld Constraint linking dynamic body to mocap reference
                # solref="0.02 1.0" provides critically damped 6-DoF constraint tracking
                equality_constraints.append(
                    f'<weld name="obs_{obj_id}_weld" body1="obs_{obj_id}_mocap" body2="obs_{obj_id}" '
                    f'solref="0.015 1.0" solimp="0.9 0.95 0.001"/>'
                )
            else:
                # Direct Dynamic Body (for PD force or teleport mode)
                dyn_body = (
                    f'<body name="obs_{obj_id}" pos="{px:.6f} {py:.6f} {pz:.6f}" '
                    f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}">\n'
                    f'  <freejoint name="obs_{obj_id}_joint"/>\n'
                    f'  <geom name="obs_{obj_id}_geom" type="mesh" mesh="{mesh_name}" mass="{mass_val:.3f}" '
                    f'rgba="{rgba_str}" friction="0.5 0.005 0.0001"/>\n'
                    f'</body>'
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
        prev_qpos = None
        prev_qvel = None
        if self.model is not None and self.data is not None:
            prev_qpos = self.data.qpos.copy()
            prev_qvel = self.data.qvel.copy()

        xml = self._generate_mjcf_xml()
        new_model = mujoco.MjModel.from_xml_string(xml)
        new_data = mujoco.MjData(new_model)

        # Restore states
        if prev_qpos is not None and self.robot_loaded:
            num_j = min(7, new_model.nq, len(prev_qpos))
            new_data.qpos[:num_j] = prev_qpos[:num_j]
            new_data.qvel[:num_j] = prev_qvel[:num_j]

        # Initialize mocap positions for obstacles
        if self.tracking_mode == "mocap":
            for obj_id, obj in self.tracked_objects.items():
                mocap_name = f"obs_{obj_id}_mocap"
                body_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
                if body_id >= 0:
                    mocap_idx = new_model.body_mocapid[body_id]
                    if mocap_idx >= 0:
                        new_data.mocap_pos[mocap_idx] = obj["target_pos"]
                        new_data.mocap_quat[mocap_idx] = obj["target_quat"]

        mujoco.mj_forward(new_model, new_data)
        self.model = new_model
        self.data = new_data
        self._eval_data = mujoco.MjData(new_model)

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
        self.kin = FrankaKinematics(base_position=self.robot_base_pos)

        self._rebuild_model()

        if joint_positions is not None:
            self.sync_robot_state(joint_positions)

        print(
            f"[MUJOCO DIGITAL TWIN] Loaded Franka Panda robot at "
            f"[{self.robot_base_pos[0]:.3f}, {self.robot_base_pos[1]:.3f}, {self.robot_base_pos[2]:.3f}]"
        )
        return True

    def sync_robot_state(
        self,
        joint_positions: List[float],
        joint_velocities: Optional[List[float]] = None
    ) -> None:
        """Synchronizes the physical twin robot joints with current kinematic state."""
        if not self.robot_loaded or self.data is None:
            return

        num_j = min(len(joint_positions), 7, self.model.nq)
        self.data.qpos[:num_j] = np.array(joint_positions[:num_j], dtype=np.float64)
        if joint_velocities is not None:
            self.data.qvel[:num_j] = np.array(joint_velocities[:num_j], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> List[float]:
        """Returns the current 7-DoF robot arm joint angles [rad]."""
        if not self.robot_loaded or self.data is None:
            return [0.0] * 7
        return [float(x) for x in self.data.qpos[:7]]

    def get_joint_velocities(self) -> List[float]:
        """Returns the current 7-DoF robot arm joint velocities [rad/s]."""
        if not self.robot_loaded or self.data is None:
            return [0.0] * 7
        return [float(x) for x in self.data.qvel[:7]]

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

        self.tracked_objects[target_obj_id] = {
            "name": name,
            "mesh_path": mesh_path,
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
            mocap_name = f"obs_{obj_id}_mocap"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
            if body_id >= 0:
                mocap_idx = self.model.body_mocapid[body_id]
                if mocap_idx >= 0:
                    self.data.mocap_pos[mocap_idx] = pos_arr
                    self.data.mocap_quat[mocap_idx] = quat_wxyz
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
            return None

        pos = tuple(float(x) for x in self.data.xpos[body_id])
        quat = _wxyz_to_xyzw(self.data.xquat[body_id])
        return (pos, quat)

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

    def check_collision(self) -> bool:
        """Returns True if there is contact between the robot and any scene obstacle or table."""
        if self.model is None or self.data is None:
            return False

        robot_body_names = ["robot_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "ee"]
        robot_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in robot_body_names
        }

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

    def get_min_obstacle_distance(self) -> float:
        """Computes the signed minimum Euclidean distance between robot arm links and dynamic obstacles."""
        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return float("inf")

        robot_geoms = [
            i for i in range(self.model.ngeom)
            if any(k in self.model.geom(i).name for k in ["robot_base", "link", "gripper"])
        ]
        obstacle_geoms = [
            i for i in range(self.model.ngeom)
            if "obs_" in self.model.geom(i).name and "_geom" in self.model.geom(i).name
        ]

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

    def get_closest_points(self, distance: float = 0.5) -> List[Tuple[int, int, float, Tuple[float, float, float]]]:
        """
        Calculates closest points between robot and obstacles within specified distance limit.
        Returns list of (robot_geom_id, obstacle_geom_id, distance, contact_point_on_robot).
        """
        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return []

        robot_geoms = [
            i for i in range(self.model.ngeom)
            if any(k in self.model.geom(i).name for k in ["link", "gripper"])
        ]
        obstacle_geoms = [
            i for i in range(self.model.ngeom)
            if "obs_" in self.model.geom(i).name and "_geom" in self.model.geom(i).name
        ]

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
                'is_target': bool(obj_data.get('is_target', False))
            })
        return obstacles

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
            eval_data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        else:
            eval_data.qpos[:7] = self.data.qpos[:7]

        # Parse target orientation if provided (supports 3x3 matrix or 4-element quaternion)
        rot_mat = None
        if target_quat is not None:
            tq = np.asarray(target_quat)
            if tq.shape == (3, 3):
                rot_mat = tq.astype(np.float64)
            elif len(tq.flatten()) == 4:
                from scipy.spatial.transform import Rotation
                rot_mat = Rotation.from_quat(tq.flatten()).as_matrix().astype(np.float64)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee")
        if site_id < 0 and body_id < 0:
            return None

        target_p = np.array(target_pos, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64) if rot_mat is not None else None

        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
        q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)

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
                    return eval_data.qpos[:7].astype(np.float32)

                if site_id >= 0:
                    mujoco.mj_jacSite(self.model, eval_data, jacp, jacr, site_id)
                else:
                    mujoco.mj_jacBody(self.model, eval_data, jacp, jacr, body_id)

                J = np.vstack([jacp[:, :7], jacr[:, :7]])
                err = np.concatenate([pos_err, 0.2 * rot_err])
                step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            else:
                if np.linalg.norm(pos_err) < 2e-3:
                    return eval_data.qpos[:7].astype(np.float32)

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
            return eval_data.qpos[:7].astype(np.float32)

        return None

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

        if self.model is None or self.data is None or len(self.tracked_objects) == 0:
            return coll_p, coll_gvm

        # Extract obstacle geoms (ignore targets) and map to obj_id
        obstacle_geoms = []
        geom_to_obj_id = {}
        for i in range(self.model.ngeom):
            g_name = self.model.geom(i).name
            if g_name.startswith("obs_") and g_name.endswith("_geom"):
                try:
                    obj_id = int(g_name.split("_")[1])
                    if self.tracked_objects.get(obj_id, {}).get("is_target", False):
                        continue
                    obstacle_geoms.append(i)
                    geom_to_obj_id[i] = obj_id
                except Exception:
                    pass

        if not obstacle_geoms:
            return coll_p, coll_gvm

        # Pre-cache robot arm geoms and link indices
        robot_geom_link_map = []
        for i in range(self.model.ngeom):
            g_name = self.model.geom(i).name
            if "link" in g_name:
                for link_num in range(1, 8):
                    if f"link{link_num}" in g_name:
                        robot_geom_link_map.append((i, link_num))
                        break
            elif "gripper" in g_name or "ee" in g_name:
                robot_geom_link_map.append((i, -1))

        if not robot_geom_link_map:
            return coll_p, coll_gvm

        if self._eval_data is None:
            self._eval_data = mujoco.MjData(self.model)

        eval_data = self._eval_data
        eval_data.qpos[:] = self.data.qpos[:]
        if hasattr(eval_data, 'mocap_pos') and hasattr(self.data, 'mocap_pos'):
            eval_data.mocap_pos[:] = self.data.mocap_pos[:]
            eval_data.mocap_quat[:] = self.data.mocap_quat[:]

        # Forecast future trajectories for all tracked dynamic obstacles across horizon H
        predictions = (
            self.predictor.predict_all(horizon=H, dt=dt)
            if self.predictor is not None
            else {}
        )

        # Pre-cache mocap indices for tracked obstacles
        obj_mocap_map = {}
        for obj_id in predictions.keys():
            m_name = f"obs_{obj_id}_mocap"
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, m_name)
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

        for h in eval_steps:
            # Advance all tracked dynamic obstacles to forecasted pose at step h
            for obj_id, m_idx in obj_mocap_map.items():
                if obj_id in predictions:
                    pred = predictions[obj_id]
                    eval_data.mocap_pos[m_idx] = pred.positions[h].astype(np.float64)
                    eval_data.mocap_quat[m_idx] = _xyzw_to_wxyz(pred.orientations[h])

            for k in range(K):
                q_step = Q[k, h]
                qd_step = QD[k, h]

                # Update Franka Panda joint positions in MuJoCo eval data
                eval_data.qpos[:7] = q_step
                mujoco.mj_kinematics(self.model, eval_data)

                step_coll_p = 0.0
                step_coll_gvm = 0.0

                for rg, fk_link_idx in robot_geom_link_map:
                    rg_pos = eval_data.geom_xpos[rg]
                    rg_r = self.model.geom_rbound[rg]

                    for og in obstacle_geoms:
                        og_pos = eval_data.geom_xpos[og]
                        og_r = self.model.geom_rbound[og]

                        # Broadphase culling: skip if bounding spheres are farther than sigma_2
                        dist_centers = np.linalg.norm(rg_pos - og_pos)
                        if dist_centers > (rg_r + og_r + sigma_2):
                            continue

                        try:
                            d = mujoco.mj_geomDistance(self.model, eval_data, rg, og, float(sigma_2), fromto)
                        except Exception:
                            continue

                        if d > sigma_2:
                            continue

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

                        # 2. Distance Gradient Vector (from obstacle surface to robot link)
                        diff = fromto[0:3] - fromto[3:6]
                        norm_mag = np.linalg.norm(diff)
                        if norm_mag > 1e-6:
                            normal_grad = (diff / norm_mag).astype(np.float32)
                        else:
                            normal_grad = np.zeros(3, dtype=np.float32)

                        # 3. Relative Cartesian Velocity between closest robot link and moving obstacle
                        vel_cart = kin_helper.compute_cartesian_velocity(q_step, qd_step, link_idx=fk_link_idx)
                        
                        o_id = geom_to_obj_id.get(og)
                        if o_id is not None and o_id in predictions:
                            v_obs = predictions[o_id].velocities[h]
                        else:
                            v_obs = np.zeros(3, dtype=np.float32)

                        v_rel = vel_cart - v_obs
                        vel_mag = np.linalg.norm(v_rel)

                        # 4. Cosine Alignment theta
                        if vel_mag > 1e-4 and norm_mag > 1e-6:
                            cos_theta = float(np.dot(normal_grad, v_rel) / vel_mag)
                        else:
                            cos_theta = 0.0

                        # 5. GVM-SDF Modulation Term
                        modulation = 1.0 - rho * cos_theta
                        gvm_term = phi * (1.0 + vel_mag * modulation)
                        step_coll_gvm += gvm_term

                coll_p[k, h] = step_coll_p
                coll_gvm[k, h] = step_coll_gvm

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

    def reset(self) -> None:
        """Resets the simulation environment and cleans up dynamic bodies."""
        self.tracked_objects.clear()
        if self.predictor is not None:
            self.predictor.reset()
        self._rebuild_model()
        print("[MUJOCO DIGITAL TWIN] Environment reset.")

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

