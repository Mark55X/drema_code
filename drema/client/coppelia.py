#!/usr/bin/env python
"""
CoppeliaSim / RLBench environment: the task scene, its vision sensors with exact segmentation, the simulated Panda.

Every loop iteration advances CoppeliaSim by exactly one physics step, so the simulated clock (not the wall clock)
is the time base for commands and frame timestamps.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..controller.franka_kinematics import FrankaKinematics
from ..visualization.sim_recorder import SimVideoRecorder, showcase_view
from .actuation import JointCommandExecutor
from .environment import Environment, RobotReading, SceneScan
from .gripper import RLBenchGripperController
from .keyframes import (KeyframeSource, PolicyKeyframeSource, StaticKeyframeSource, TaskKeyframeSource,
                        keyframes_from_demo)

try:
    from rlbench.environment import Environment as RLBenchEnvironment
    from rlbench.observation_config import ObservationConfig
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.tasks import DynamicDremaTest1, ReachTargetMovingCube
    HAS_RLBENCH = True
except Exception as e:
    print(f"[Warning] Could not import RLBench: {e}")
    HAS_RLBENCH = False


class CoppeliaEnvironment(Environment):

    client_tag = "CoppeliaClient"
    log_tag = "COPPELIA ENVIRONMENT"
    sync_modes = ("stepped", "realtime")
    keyframe_modes = ("auto", "task", "demo", "policy", "none")

    SCAN_MAX_RECREATIONS = 3

    def __init__(
        self,
        task_name: str = "dynamic_drema_test_1",
        headless: bool = False,
        reachability_radius: float = 0.95,
        scan_resolution: Tuple[int, int] = (1280, 720),
        label_non_physical_shapes: bool = False,
        cam_resolution: Tuple[int, int] = (256, 256),
        record_dir: Optional[str] = None,
        record_camera: str = "showcase",
        record_resolution: Tuple[int, int] = (960, 540),
        record_every: int = 1,
        record_fov: Optional[float] = None,
        record_eye: Optional[List[float]] = None,
        record_look_at: Optional[List[float]] = None,
        actuation: str = "position"
    ):
        """:param actuation: How the MPC joint velocities reach the arm, see drema.client.actuation."""
        self.task_name = task_name
        self.headless = headless
        self.reachability_radius = reachability_radius
        self.scan_resolution = tuple(scan_resolution)
        self.label_non_physical_shapes = bool(label_non_physical_shapes)
        self.cam_resolution = tuple(cam_resolution)
        self.record = dict(dir=record_dir, camera=record_camera, resolution=tuple(record_resolution),
                           every=record_every, fov=record_fov, eye=record_eye, look_at=record_look_at)
        self.actuation_mode = actuation
        self.semantic_labels: Dict[str, int] = {}
        self.label_categories: Dict[int, str] = {}
        self.object_physics: Dict[int, Dict[str, Any]] = {}
        self.task_description = ""
        self.video_recorder = None
        self.sim_steps = 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def launch(self) -> None:
        if not HAS_RLBENCH:
            raise RuntimeError("RLBench and PyRep must be installed to run the CoppeliaSim environment.")

        obs_config = ObservationConfig()
        obs_config.front_camera.rgb = True
        obs_config.front_camera.depth = True
        obs_config.front_camera.image_size = self.cam_resolution

        obs_config.wrist_camera.rgb = True
        obs_config.wrist_camera.depth = True
        obs_config.wrist_camera.image_size = self.cam_resolution

        obs_config.overhead_camera.rgb = True
        obs_config.overhead_camera.depth = True
        obs_config.overhead_camera.image_size = self.cam_resolution

        action_mode = ActionMode(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
        self.env = RLBenchEnvironment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=self.headless,
            robot_setup='panda'
        )
        self.env.launch()

        if self.task_name == "dynamic_drema_test_1":
            task_cls = DynamicDremaTest1
        elif self.task_name == "reach_target_moving_cube":
            task_cls = ReachTargetMovingCube
        else:
            from rlbench import tasks
            task_cls = getattr(tasks, self.task_name, None)
            if not isinstance(task_cls, type):
                # snake_case task file name, as in the RLBench / PerAct task lists (e.g. close_jar)
                from rlbench.backend.utils import task_file_to_task_class
                task_cls = task_file_to_task_class(self.task_name)

        self.task = self.env.get_task(task_cls)
        descriptions, self.current_obs = self.task.reset()
        print(f"✓ Loaded RLBench Task: {self.task_name}")
        print(f"  Task description: {descriptions[0]}")
        self.task_description = descriptions[0]

        scene = self.task._scene
        self.cameras = {
            'front': scene._cam_front,
            'wrist': scene._cam_wrist,
            'overhead': scene._cam_overhead
        }
        self.mask_cameras = {
            'front': getattr(scene, '_cam_front_mask', None),
            'wrist': getattr(scene, '_cam_wrist_mask', None),
            'overhead': getattr(scene, '_cam_overhead_mask', None)
        }

        self.gripper = RLBenchGripperController(self.task._robot, self.task._task)
        self._scene_shape_labels()

    def ready(self) -> None:
        rec = self.record
        if rec['dir']:
            eye, look_at, reference = rec['eye'], rec['look_at'], self.cameras.get(rec['camera'])
            if rec['camera'] == "showcase" and (eye is None or look_at is None):
                task_inst = getattr(self.task, '_task', self.task)
                target = task_inst.target.get_position() if hasattr(task_inst, 'target') else [0.4, 0.0, 0.8]
                default_eye, default_look = showcase_view(self.task._robot.arm.joints[0].get_position(), target)
                eye = default_eye if eye is None else eye
                look_at = default_look if look_at is None else look_at
            self.video_recorder = SimVideoRecorder(
                os.path.join(rec['dir'], time.strftime("%Y%m%d_%H%M%S")), reference,
                resolution=rec['resolution'], every_n=rec['every'],
                view_angle_deg=rec['fov'] if rec['fov'] is not None else (60.0 if eye is not None else None),
                eye=eye, look_at=look_at
            )

        self.dt = float(self.env._pyrep.get_simulation_timestep())
        self.actuator = JointCommandExecutor(self.task._robot.arm, self.actuation_mode, dt=self.dt)
        self.actuator.reset()
        print(f"[{self.client_tag}] Arm actuation: {self.actuation_mode}")

    def reset(self) -> None:
        self.sim_steps = 0
        descriptions, self.current_obs = self.task.reset()
        self.task_description = descriptions[0]

    def shutdown(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
        if hasattr(self, 'env'):
            try:
                self.env.shutdown()
            except Exception:
                pass

    def describe(self) -> None:
        arm = self.task._robot.arm
        try:
            tip_rel_j7 = arm.get_tip().get_position(relative_to=arm.joints[6])
            tip_rel_arm = arm.get_tip().get_position(relative_to=arm)
            print("\n" + "=" * 60)
            print("  [KINEMATICS INSPECTION FROM COPPELIASIM]")
            print(f"  • Robot model root pos (arm.get_position()):      {arm.get_position().round(5).tolist()}")
            print(f"  • Robot model root ori (arm.get_orientation()):   {np.array(arm.get_orientation()).round(5).tolist()}")
            print(f"  • Joint 1 world pos (arm.joints[0]):              {arm.joints[0].get_position().round(5).tolist()}")
            print(f"  • Joint 1 world ori (arm.joints[0]):              {np.array(arm.joints[0].get_orientation()).round(5).tolist()}")
            print(f"  • Joint 7 world pos (arm.joints[6]):              {arm.joints[6].get_position().round(5).tolist()}")
            print(f"  • Joint 7 world ori (arm.joints[6]):              {np.array(arm.joints[6].get_orientation()).round(5).tolist()}")
            print(f"  • Fingertip world pos (arm.get_tip()):           {arm.get_tip().get_position().round(5).tolist()}")
            print(f"  • Fingertip world ori (arm.get_tip()):           {np.array(arm.get_tip().get_orientation()).round(5).tolist()}")
            print(f"  • Fingertip offset relative to Joint 7:           {tip_rel_j7.round(5).tolist()}")
            print(f"  • Fingertip offset relative to Arm Root:          {tip_rel_arm.round(5).tolist()}")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[KINEMATICS INSPECTION ERROR] {e}")

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return FrankaKinematics.Q_MIN.astype(np.float64), FrankaKinematics.Q_MAX.astype(np.float64)

    def timestamp(self) -> float:
        # Shifted by one step because the server treats timestamp 0 as missing;
        # the Kalman predictor only uses timestamp differences.
        return (self.sim_steps + 1) * self.dt

    # -------------------------------------------------------------------------
    # Sensing
    # -------------------------------------------------------------------------

    def capture_scan(self, num_steps: int) -> SceneScan:
        """
        Dense 360-degree orbital scan at t=0 while the simulation physics is frozen:
        4 orbital cameras x num_steps rotation steps.
        """
        from pyrep.objects.dummy import Dummy
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode

        print(f"[{self.client_tag}] 360° orbital scan at t=0 ({num_steps * 4} views). Physics is stationary.")

        if not Dummy.exists('cam_cinematic_placeholder'):
            placeholder = Dummy.create()
            placeholder.set_name('cam_cinematic_placeholder')
            placeholder.set_position([0.25, 0.0, 0.75])
        else:
            placeholder = Dummy('cam_cinematic_placeholder')

        if not Dummy.exists('cam_cinematic_base'):
            base_dummy = Dummy.create()
            base_dummy.set_name('cam_cinematic_base')
            base_dummy.set_position([0.25, 0.0, 0.75])
        else:
            base_dummy = Dummy('cam_cinematic_base')

        placeholder.set_parent(base_dummy)

        # Base camera pose from the front camera (facing the workspace)
        base_pose = list(self.cameras['front'].get_pose())

        # 4 cameras at distinct vertical and radial offsets (identical to prepare_data_for_drema)
        cams = []
        cams_mask = []
        resolutions = list(self.scan_resolution)
        p_offsets = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.30],
            [0.0, 0.0, -0.50],
            [0.0, 0.50, -0.25]
        ]

        def create_sensor(pose, render_mode):
            s = VisionSensor.create(
                resolution=resolutions,
                explicit_handling=True,
                near_clipping_plane=0.01,
                far_clipping_plane=4.5,
                view_angle=40.0,
                render_mode=render_mode
            )
            s.set_pose(pose)
            s.set_parent(placeholder)
            return s

        for idx in range(4):
            p = list(base_pose)
            p[0] += p_offsets[idx][0]
            p[1] += p_offsets[idx][1]
            p[2] += p_offsets[idx][2]
            cams.append(create_sensor(p, RenderMode.OPENGL3))
            cams_mask.append(create_sensor(p, RenderMode.OPENGL_COLOR_CODED))

        rotate_angle = (2.0 * np.pi) / float(num_steps)
        all_scan_cameras = []

        try:
            for step_idx in range(num_steps):
                base_dummy.rotate([0, 0, rotate_angle])

                for c_idx, (c, c_mask) in enumerate(zip(cams, cams_mask)):
                    # In CoppeliaSim, explicit_handling requires handle_explicit() before capture
                    c.handle_explicitly()
                    c_mask.handle_explicitly()
                    rgb_float = c.capture_rgb()
                    # Headless CoppeliaSim can leave a sensor rendering black (and garbage depth) for good while a
                    # fresh one at the same pose renders fine from its second frame
                    for attempt in range(1, self.SCAN_MAX_RECREATIONS + 1):
                        if rgb_float.mean() * 255.0 >= 1.0:
                            break
                        print(f"[{self.client_tag} Scan] Camera {c_idx} renders black at step {step_idx}: "
                              f"recreating it ({attempt}/{self.SCAN_MAX_RECREATIONS})")
                        pose = c.get_pose()
                        c.remove()
                        c = cams[c_idx] = create_sensor(pose, RenderMode.OPENGL3)
                        c.handle_explicitly()
                        c.capture_rgb()
                        c.handle_explicitly()
                        rgb_float = c.capture_rgb()
                    if rgb_float.mean() * 255.0 < 1.0:
                        print(f"[{self.client_tag} Scan] Camera {c_idx} still black at step {step_idx}: view not sent")
                        continue
                    rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                    depth_m = c.capture_depth(in_meters=True).astype(np.float32)
                    ext = np.array(c.get_matrix(), dtype=np.float32).reshape((4, 4))
                    intrinsic = self._pixel_centre_intrinsics(c)
                    near_clip = float(c.get_near_clipping_plane()) if hasattr(c, 'get_near_clipping_plane') else 0.01
                    far_clip = float(c.get_far_clipping_plane()) if hasattr(c, 'get_far_clipping_plane') else 3.5

                    # Exact integer handle per pixel from the color-coded mask
                    m_rgb = c_mask.capture_rgb()
                    m_255 = (np.clip(m_rgb, 0.0, 1.0) * 255.0).astype(int)
                    mask_int32 = (m_255[:, :, 0] + m_255[:, :, 1] * 256 + m_255[:, :, 2] * 256 * 256).astype(np.int32)

                    cam_name = f'orbit_{step_idx}_{c_idx}'
                    all_scan_cameras.append((cam_name, rgb_uint8, depth_m, ext, intrinsic, near_clip, far_clip, mask_int32))

                if (step_idx + 1) % 10 == 0:
                    print(f"[{self.client_tag} Scan] Captured {(step_idx + 1) * 4}/{num_steps * 4} views...")

            self._scene_shape_labels()
            joint_positions = []
            try:
                joint_positions = list(self.task._robot.arm.get_joint_positions())
            except Exception:
                pass
            return SceneScan(views=all_scan_cameras, joint_positions=joint_positions)
        finally:
            for c in cams:
                try:
                    c.remove()
                except Exception:
                    pass
            for c_mask in cams_mask:
                try:
                    c_mask.remove()
                except Exception:
                    pass
            try:
                base_dummy.set_orientation([0, 0, 0])
            except Exception:
                pass

    def _scene_shape_labels(self) -> Dict[str, int]:
        """
        CoppeliaSim shape name -> handle; handles are the object ids shared with perception and the roles.
        Shapes outside the physics (spawn areas, painted targets, success zones) get no label unless
        label_non_physical_shapes: perception then keeps their pixels as static background, never as objects.
        Each label gets its category from the scene tree (self.label_categories), so perception needs no names: the
        robot's shapes are 'robot', the task's 'object', the rest of the scene (table, walls) 'background'.
        """
        from pyrep.backend import sim
        from pyrep.const import ObjectType
        from pyrep.objects.shape import Shape
        robot = self.task._robot
        robot_handles = {o.get_handle() for part in (robot.arm, robot.gripper)
                         for o in [part] + part.get_objects_in_tree(object_type=ObjectType.SHAPE)}
        task_base = getattr(self.task, '_task', self.task).get_base()
        task_handles = {o.get_handle() for o in task_base.get_objects_in_tree(object_type=ObjectType.SHAPE)}
        handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
        labels, categories, physics = {}, {}, {}
        filter_names = ["DefaultCamera", "ResizableFloor", "Floor", "Wall", "Ceiling"]
        for h in handles:
            try:
                shape = Shape(h)
                name = shape.get_name()
                if any(f in name for f in filter_names):
                    continue
                if not self.label_non_physical_shapes and not self._is_physical(shape):
                    continue
                labels[name] = int(h)
                categories[int(h)] = ('robot' if h in robot_handles else
                                      'object' if h in task_handles else 'background')
                if categories[int(h)] == 'object':
                    physics[int(h)] = self._shape_physics(shape)
            except Exception:
                pass
        self.semantic_labels = labels
        self.label_categories = categories
        self.object_physics = {h: p for h, p in physics.items() if p}
        return labels

    @staticmethod
    def _shape_physics(shape) -> Dict[str, Any]:
        """Mass, Bullet friction and static flag of the respondable shape a labelled shape belongs to ({} if none)."""
        from pyrep.backend import sim
        from pyrep.objects.shape import Shape
        node = shape
        while node is not None:
            handle = node.get_handle()
            if sim.simGetObjectType(handle) == sim.sim_object_shape_type and Shape(handle).is_respondable():
                body = Shape(handle)
                try:
                    return {'mass': float(body.get_mass()), 'friction': float(body.get_bullet_friction()),
                            'static': not body.is_dynamic()}
                except Exception:
                    return {}
            node = node.get_parent()
        return {}

    @staticmethod
    def _pixel_centre_intrinsics(sensor) -> np.ndarray:
        """
        Intrinsics for pixels indexed from 0: PyRep puts the principal point at (W/2, H/2) instead of
        ((W-1)/2, (H-1)/2), which shifts every back-projected point by half a pixel (2 mm at 1.5 m on a 256 px camera).
        """
        K = np.array(sensor.get_intrinsic_matrix(), dtype=np.float32)
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        return K

    @staticmethod
    def _is_physical(shape) -> bool:
        """The shape or one of its ancestor shapes is respondable (a visual mesh belongs to its respondable parent)."""
        from pyrep.backend import sim
        from pyrep.objects.shape import Shape
        node = shape
        while node is not None:
            handle = node.get_handle()
            if sim.simGetObjectType(handle) == sim.sim_object_shape_type and Shape(handle).is_respondable():
                return True
            node = node.get_parent()
        return False

    def capture_cameras(self) -> Dict[str, Dict[str, Any]]:
        """RGB, depth, extrinsics, intrinsics and masks of the streaming vision sensors."""
        cam_dict = {}
        for name, cam in self.cameras.items():
            try:
                if cam is not None:
                    try:
                        cam.handle_explicitly()
                    except Exception:
                        pass

                rgb_float = cam.capture_rgb()
                rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                depth_m = cam.capture_depth(in_meters=True).astype(np.float32)
                ext = np.array(cam.get_matrix(), dtype=np.float32).reshape((4, 4))
                intrinsic = self._pixel_centre_intrinsics(cam)
                near_clip = float(cam.get_near_clipping_plane()) if hasattr(cam, 'get_near_clipping_plane') else 0.01
                far_clip = float(cam.get_far_clipping_plane()) if hasattr(cam, 'get_far_clipping_plane') else 3.5

                mask_cam = self.mask_cameras.get(name)
                mask_int32 = None
                if mask_cam is not None:
                    try:
                        mask_cam.handle_explicitly()
                        m_rgb = mask_cam.capture_rgb()
                        m_255 = (np.clip(m_rgb, 0.0, 1.0) * 255.0).astype(int)
                        mask_int32 = (m_255[:, :, 0] + m_255[:, :, 1] * 256 + m_255[:, :, 2] * 256 * 256).astype(np.int32)
                    except Exception:
                        mask_int32 = None

                cam_dict[name] = {
                    'rgb': rgb_uint8,
                    'depth': depth_m,
                    'extrinsics': ext,
                    'intrinsics': intrinsic,
                    'near_clipping': near_clip,
                    'far_clipping': far_clip,
                    'mask': mask_int32
                }
            except Exception:
                continue
        return cam_dict

    def robot_base_position(self) -> List[float]:
        """
        World position of the panda_link0 base frame. In CoppeliaSim, arm.get_position() returns the pedestal
        root [-0.309, 0.0, 0.820], while the kinematic chain starts at the tabletop mount, d1 = 0.333 m below
        joint 1: base_z = j1_z - 0.333 m = 0.750 m.
        """
        arm = getattr(getattr(self, 'task', None), '_robot', None)
        arm = getattr(arm, 'arm', None)
        if arm is not None and hasattr(arm, 'joints') and len(arm.joints) > 0:
            try:
                j1 = arm.joints[0].get_position()
                return [float(j1[0]), float(j1[1]), float(j1[2] - 0.333)]
            except Exception:
                pass
        if arm is not None and hasattr(arm, 'get_position'):
            try:
                return [float(x) for x in arm.get_position()]
            except Exception:
                pass
        return [0.0, 0.0, 0.0]

    def read_robot(self) -> RobotReading:
        robot = self.task._robot
        return RobotReading(
            joint_positions=list(robot.arm.get_joint_positions()),
            joint_velocities=list(robot.arm.get_joint_velocities()),
            ee_pose=robot.arm.get_tip().get_pose().tolist(),
            gripper_open=float(robot.gripper.get_open_amount()[0])
        )

    # -------------------------------------------------------------------------
    # Task
    # -------------------------------------------------------------------------

    def keyframe_source(self, mode: str, policy_spec: Optional[str]) -> Optional[KeyframeSource]:
        task_inst = getattr(self.task, '_task', self.task)
        if mode == "auto":
            mode = "policy" if policy_spec else ("task" if hasattr(task_inst, 'get_keyframes') else "none")

        if mode == "task":
            return TaskKeyframeSource(task_inst)
        if mode == "demo":
            demo = None
            for attempt in range(3):
                try:
                    # The RLBench fork calls callable_each_step(reset=True) after a failed attempt: it must not be None.
                    demo = self.task.get_demos(1, live_demos=True, callable_each_step=lambda *a, **k: None)[0]
                    break
                except Exception as e:
                    print(f"[{self.client_tag}] [Warning] Demo recording failed (attempt {attempt + 1}/3): {e}")
            if demo is None:
                raise RuntimeError(f"Could not record a demo for task '{self.task_name}'")
            frames = keyframes_from_demo(demo)
            descriptions, self.current_obs = self.task.reset_to_demo(demo)
            self.task_description = descriptions[0]
            print(f"[{self.client_tag}] Demo keyframes (PerAct keypoint discovery): {len(frames)} -> "
                  + ", ".join(f"{k.name}[grip={'open' if k.gripper_open else 'close'}"
                              f"{', ignore_coll' if k.ignore_collisions else ''}]" for k in frames))
            return StaticKeyframeSource(frames)
        if mode == "policy":
            return PolicyKeyframeSource.load(policy_spec, self)
        return None

    def fallback_target(self) -> Tuple[List[float], bool]:
        task_inst = getattr(self.task, '_task', self.task)
        if hasattr(task_inst, 'get_target_ee_pose'):
            return list(task_inst.get_target_ee_pose()), True
        if hasattr(task_inst, 'target'):
            pos = list(task_inst.target.get_position())
            if getattr(task_inst, 'target_ee_orientation', None) is not None:
                return pos + list(task_inst.target_ee_orientation), True
            # Position-only goal if no EE orientation specified by task
            return pos, True
        return [], False

    def observe(self) -> Any:
        """Current RLBench Observation (for keyframe policies such as PerAct)."""
        return self.task._scene.get_observation()

    def advance(self, step: int, task_active: bool) -> None:
        self.gripper.step()
        self.task._task.step()  # Moves the task's scripted obstacles (oscillating tunnel)
        self.env._pyrep.step()
        self.sim_steps += 1
        if self.video_recorder is not None:
            self.video_recorder.capture(step, self.sim_steps * self.dt, task_active)

    def check_success(self) -> bool:
        success, _ = self.task._task.success()
        return bool(success)
