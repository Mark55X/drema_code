#!/usr/bin/env python
"""
CoppeliaSim / RLBench Closed-Loop Client (run_coppelia_client.py)

Controls the simulation environment:
- Runs the task (default: dynamic_drema_test_1 with oscillating tunnel obstacle).
- Immediately begins multi-camera RGB-D streaming to DREMA upon start.
- Interactive CLI control: type 'start' to engage task execution, 'stop'/'pause' to halt, 'reset' to restart, 'quit' to exit.
- Supports dual frequency: f_cam ≈ 10-15 Hz for perception streaming, f_ctrl ≈ 50 Hz for joint velocity control.
- Supports both --sync_mode stepped (lockstep) and --sync_mode realtime (wall-clock continuous).
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# Ensure paths are set (do not shadow installed PyRep with source tree)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../RLBench")))

from drema.communication.grpc_client import DremaGrpcClient

try:
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.tasks import DynamicDremaTest1, ReachTargetMovingCube
    HAS_RLBENCH = True
except Exception as e:
    print(f"[Warning] Could not import RLBench: {e}")
    HAS_RLBENCH = False


class CoppeliaSimulationClient:
    """
    Manages CoppeliaSim simulation, sensor streaming, and interactive CLI control.
    """

    def __init__(
        self,
        server_address: str = "localhost:50051",
        task_name: str = "dynamic_drema_test_1",
        sync_mode: str = "realtime",
        headless: bool = False,
        cam_fps: float = 10.0,
        ctrl_fps: float = 50.0,
        reachability_radius: float = 0.95,
        scan_resolution: Tuple[int, int] = (1280, 720),
        scan_steps: int = 50,
        scan_chunk_size: int = 2,
        scan_chunk_timeout: float = 180.0,
        ping_timeout: float = 1.5,
        ping_max_retries: int = 3,
        force_scan: bool = False,
        cam_resolution: Tuple[int, int] = (256, 256),
        log_interval_actions: int = 10
    ):
        self.server_address = server_address
        self.task_name = task_name
        self.sync_mode = sync_mode.lower()
        self.headless = headless
        self.cam_fps = cam_fps
        self.ctrl_fps = ctrl_fps
        self.reachability_radius = reachability_radius
        self.scan_resolution = tuple(scan_resolution)
        self.scan_steps = int(scan_steps)
        self.scan_chunk_size = int(scan_chunk_size)
        self.scan_chunk_timeout = float(scan_chunk_timeout)
        self.ping_timeout = float(ping_timeout)
        self.ping_max_retries = int(ping_max_retries)
        self.force_scan = bool(force_scan)
        self.cam_resolution = tuple(cam_resolution)
        self.log_interval_actions = max(0, int(log_interval_actions))

        self.cam_period = 1.0 / max(1.0, cam_fps)
        self.ctrl_period = 1.0 / max(1.0, ctrl_fps)
        self.cam_decimation = max(1, int(round(self.ctrl_fps / self.cam_fps)))

        # Interactive state
        self.task_active = False
        self.running = True
        self.step_counter = 0
        self.active_actions_count = 0
        self._reset_requested = False
        self.server_connected = False
        self.initial_scan_done = False
        self._last_ping_time = 0.0
        self._failed_pings = 0

        # Initialize gRPC Client
        print(f"[CoppeliaClient] Connecting to DREMA suite at {self.server_address}...")
        self.client = DremaGrpcClient(target_address=self.server_address)
        is_alive, status = self.client.ping_status(timeout=self.ping_timeout)
        self.server_connected = is_alive
        if self.server_connected:
            print(f"✓ Connected to DREMA Dynamic Inference Suite! (Status: {status})")
            if status == "SCAN_READY":
                print("✓ DREMA server already has scene initialized (from disk cache or previous scan)! Ready for dynamic streaming.")
                self.initial_scan_done = True
        else:
            print(f"[Notice] DREMA Suite server not responding yet at {self.server_address}. Simulation will run and connect automatically as soon as it goes online.")

        # Start non-blocking camera streaming worker
        self.client.start_streaming()

        # Initialize RLBench Environment
        self._init_rlbench()

        if self.force_scan:
            self.initial_scan_done = False

        # Start interactive CLI listener thread
        self.cli_thread = threading.Thread(target=self._cli_listener, daemon=True)
        self.cli_thread.start()

    def perform_initial_scan(self, num_steps: int = 50) -> bool:
        """
        Performs dense 360-degree orbital scanning at t=0 while the simulation physics is frozen.
        Captures 4 orbital cameras x 50 steps = 200 views and streams them to DREMA suite.
        """
        from pyrep.objects.dummy import Dummy
        from pyrep.objects.vision_sensor import VisionSensor
        from pyrep.const import RenderMode

        print("\n=======================================================")
        print(f"[CoppeliaClient] Performing 360° orbital scan at t=0 ({num_steps * 4} views)...")
        print("                 Physics is stationary. Reconstructing table & obstacles...")
        print("=======================================================\n")

        # Setup placeholder and base dummy
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

        # Base camera pose from front camera (facing workspace)
        cam_front = self.cameras['front']
        base_pose = list(cam_front.get_pose())

        # Setup 4 cameras at distinct vertical and radial offsets (identical to prepare_data_for_drema)
        cams = []
        cams_mask = []
        resolutions = list(self.scan_resolution)
        p_offsets = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.30],
            [0.0, 0.0, -0.50],
            [0.0, 0.50, -0.25]
        ]
        for idx in range(4):
            c = VisionSensor.create(
                resolution=resolutions,
                explicit_handling=True,
                near_clipping_plane=0.01,
                far_clipping_plane=4.5,
                view_angle=40.0,
                render_mode=RenderMode.OPENGL3
            )
            c_mask = VisionSensor.create(
                resolution=resolutions,
                explicit_handling=True,
                near_clipping_plane=0.01,
                far_clipping_plane=4.5,
                view_angle=40.0,
                render_mode=RenderMode.OPENGL_COLOR_CODED
            )
            p = list(base_pose)
            p[0] += p_offsets[idx][0]
            p[1] += p_offsets[idx][1]
            p[2] += p_offsets[idx][2]
            c.set_pose(p)
            c_mask.set_pose(p)
            c.set_parent(placeholder)
            c_mask.set_parent(placeholder)
            cams.append(c)
            cams_mask.append(c_mask)

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
                    rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                    depth_m = c.capture_depth(in_meters=True).astype(np.float32)
                    ext = np.array(c.get_matrix(), dtype=np.float32).reshape((4, 4))
                    intrinsic = np.array(c.get_intrinsic_matrix(), dtype=np.float32)
                    near_clip = float(c.get_near_clipping_plane()) if hasattr(c, 'get_near_clipping_plane') else 0.01
                    far_clip = float(c.get_far_clipping_plane()) if hasattr(c, 'get_far_clipping_plane') else 3.5

                    # Extract exact integer handle per pixel from color-coded mask
                    m_rgb = c_mask.capture_rgb()
                    m_255 = (np.clip(m_rgb, 0.0, 1.0) * 255.0).astype(int)
                    mask_int32 = (m_255[:, :, 0] + m_255[:, :, 1] * 256 + m_255[:, :, 2] * 256 * 256).astype(np.int32)

                    cam_name = f'orbit_{step_idx}_{c_idx}'
                    all_scan_cameras.append((cam_name, rgb_uint8, depth_m, ext, intrinsic, near_clip, far_clip, mask_int32))

                if (step_idx + 1) % 10 == 0:
                    print(f"[CoppeliaClient Scan] Captured {(step_idx + 1) * 4}/{num_steps * 4} views...")

            # Extract semantic labels from CoppeliaSim scene shapes
            from pyrep.backend import sim
            from pyrep.objects.shape import Shape
            handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
            semantic_labels = {}
            filter_names = ["DefaultCamera", "ResizableFloor", "Floor", "Wall", "Ceiling"]
            for h in handles:
                try:
                    name = Shape(h).get_name()
                    if any(f in name for f in filter_names):
                        continue
                    semantic_labels[name] = int(h)
                except Exception:
                    pass

            print(f"[CoppeliaClient Scan] Extracted {len(semantic_labels)} semantic labels from CoppeliaSim scene.")

            # Retrieve real robot base position and initial joint positions
            robot_base_pos = self.get_calibrated_robot_base_pos()
            robot_joint_positions = []
            if hasattr(self, 'task') and hasattr(self.task, '_robot'):
                try:
                    robot_joint_positions = list(self.task._robot.arm.get_joint_positions())
                except Exception:
                    pass

            # Send views in robust chunks with automatic retry
            res = self.client.push_initial_scan_batch(
                all_scan_cameras,
                semantic_labels=semantic_labels,
                robot_base_pos=robot_base_pos,
                reachability_radius=self.reachability_radius,
                joint_positions=robot_joint_positions,
                chunk_size=self.scan_chunk_size,
                chunk_timeout_s=self.scan_chunk_timeout
            )

            if res and res.initial_scan_ready:
                print(f"\n✓ [CoppeliaClient] 360° orbital scan complete! All {len(all_scan_cameras)} views sent. DREMA scene populated.")
                self.initial_scan_done = True
                return True
            else:
                print(f"\n❌ [CoppeliaClient Error] DREMA server failed to acknowledge initial scan (status: {res})!")
                print(f"   Dynamic tracking cannot proceed until the scene is initialized. Type 'scan' or 'reset' to retry.")
                self.initial_scan_done = False
                return False

        except Exception as e:
            print(f"[CoppeliaClient Error] Orbital scan failed: {e}")
            return False
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

    def _init_rlbench(self):
        if not HAS_RLBENCH:
            raise RuntimeError("RLBench and PyRep must be installed to run CoppeliaSimulationClient.")

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
        self.env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=self.headless,
            robot_setup='panda'
        )
        self.env.launch()

        # Resolve Task Class
        if self.task_name == "dynamic_drema_test_1":
            task_cls = DynamicDremaTest1
        elif self.task_name == "reach_target_moving_cube":
            task_cls = ReachTargetMovingCube
        else:
            from rlbench import tasks
            task_cls = getattr(tasks, self.task_name)

        self.task = self.env.get_task(task_cls)
        descriptions, self.current_obs = self.task.reset()
        print(f"✓ Loaded RLBench Task: {self.task_name}")
        print(f"  Task description: {descriptions[0]}")

        # Cache sensor references from scene
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

    def _cli_listener(self):
        """CLI thread listening for interactive commands."""
        time.sleep(0.5)
        print("\n=======================================================")
        print("  CoppeliaSim Client Interactive Console Ready!       ")
        print("  Commands:                                           ")
        print("    'start' (or press Enter) -> Begin task execution  ")
        print("    'stop'  (or 'pause')     -> Halt robot motion     ")
        print("    'reset'                  -> Reset episode         ")
        print("    'quit'  (or 'exit')      -> Exit simulation       ")
        print("=======================================================\n")

        while self.running:
            try:
                cmd = sys.stdin.readline()
                if not cmd:
                    break
                cmd = cmd.strip().lower()

                if cmd in ['start', 'run']:
                    self.task_active = True
                    print(f"\n[COPPELIA ENVIRONMENT] >>> TASK STARTED! Robot closed-loop control engaged (f_ctrl={self.ctrl_fps}Hz).\n")
                    if not self.server_connected:
                        print("[COPPELIA ENVIRONMENT] [Notice] Waiting for DREMA suite connection before actuating robot...\n")
                elif cmd in ['scan', 's']:
                    print("\n[COPPELIA ENVIRONMENT] >>> Initiating initial scene scan...")
                    self.perform_initial_scan(num_steps=self.scan_steps)
                elif cmd in ['stop', 'pause', 'halt']:
                    self.task_active = False
                    print("\n[COPPELIA ENVIRONMENT] >>> TASK PAUSED! Robot holding position (streaming continues).\n")
                elif cmd in ['reset', 'r']:
                    self.task_active = False
                    self._reset_requested = True
                    print("\n[COPPELIA ENVIRONMENT] >>> Reset requested. Will reset cleanly on next simulation tick.\n")
                elif cmd in ['quit', 'exit', 'q']:
                    print("\n[COPPELIA ENVIRONMENT] >>> Quitting simulation...")
                    self.running = False
                    break
                else:
                    print(f"[COPPELIA ENVIRONMENT] Unknown command '{cmd}'. Type 'start', 'stop', 'reset', or 'quit'.")
            except Exception:
                break

    def capture_camera_data(self):
        """Extracts RGB, Depth, Extrinsics, Intrinsics, and Masks from all vision sensors."""
        cam_dict = {}
        for name, cam in self.cameras.items():
            try:
                if cam is not None:
                    try:
                        cam.handle_explicitly()
                    except Exception:
                        pass

                # RGB: uint8 [0, 255]
                rgb_float = cam.capture_rgb()
                rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

                # Depth: float32 in meters
                depth_m = cam.capture_depth(in_meters=True).astype(np.float32)

                # 4x4 Cam-to-world pose
                ext = np.array(cam.get_matrix(), dtype=np.float32).reshape((4, 4))

                # 3x3 Intrinsic matrix
                intrinsic = np.array(cam.get_intrinsic_matrix(), dtype=np.float32)

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
                if not self.running:
                    return cam_dict
                continue
        return cam_dict

    def get_calibrated_robot_base_pos(self) -> List[float]:
        """
        Computes true world origin coordinates of panda_link0 base frame [x, y, z].
        In CoppeliaSim, arm.get_position() returns the pedestal bounding root [-0.309, 0.0, 0.820].
        However, the physical robot kinematic chain starts at the tabletop mount where
        Joint 1 is positioned at d1 = 0.333m above panda_link0.
        Thus: base_z = j1_z - 0.333m = 0.750m (exact tabletop level).
        """
        if hasattr(self, 'task') and hasattr(self.task, '_robot'):
            arm = getattr(self.task._robot, 'arm', None)
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

    def run(self):
        """Main execution loop balancing sensing and control rates."""
        print(f"[CoppeliaClient] Simulation loop started (Sync Mode: {self.sync_mode}).")
        print("[CoppeliaClient] Multi-camera streaming active immediately. Robot is IDLE until 'start' command.")

        robot = self.task._robot
        arm = robot.arm
        gripper = robot.gripper

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

        # If server is already online at launch, perform initial scan if needed
        if self.server_connected and not self.initial_scan_done:
            self.perform_initial_scan(num_steps=self.scan_steps)

        try:
            while self.running:
                loop_start = time.time()

                # Handle asynchronous reset request from CLI safely on the main thread
                if self._reset_requested:
                    self._reset_requested = False
                    self.task_active = False
                    self.step_counter = 0
                    self.active_actions_count = 0
                    self.initial_scan_done = False
                    if self.server_connected:
                        try:
                            self.client.reset_episode(episode_index=0, task_name=self.task_name, timeout=0.5)
                        except Exception:
                            pass
                    descriptions, self.current_obs = self.task.reset()
                    print("\n[COPPELIA ENVIRONMENT] >>> EPISODE RESET COMPLETE! Re-scanning initial scene...\n")
                    if self.server_connected:
                        self.perform_initial_scan(num_steps=self.scan_steps)
                    print("\n[COPPELIA ENVIRONMENT] >>> Ready. Press 'start' to resume dynamic task.\n")
                    continue

                # Periodic non-blocking connection check to DREMA suite
                if loop_start - self._last_ping_time > 1.5:
                    self._last_ping_time = loop_start
                    is_alive, status = self.client.ping_status(timeout=self.ping_timeout)
                    if is_alive:
                        self._failed_pings = 0
                        if not self.server_connected:
                            self.server_connected = True
                            print(f"\n✓ [CoppeliaClient] Connected to DREMA Dynamic Inference Suite! (Status: {status})\n")
                            if status == "SCAN_READY":
                                print("✓ [CoppeliaClient] DREMA server already has scene initialized! Ready for dynamic streaming.\n")
                                self.initial_scan_done = True
                            elif not self.initial_scan_done:
                                self.perform_initial_scan(num_steps=self.scan_steps)
                    else:
                        self._failed_pings += 1
                        if self._failed_pings >= self.ping_max_retries and self.server_connected:
                            self.server_connected = False
                            print(f"\n[Notice] [CoppeliaClient] DREMA Suite disconnected (missed {self.ping_max_retries} consecutive pings). Holding position.\n")

                self.step_counter += 1

                # 1. Perception Step (f_cam ≈ 10 Hz): capture and push frames only when scene is initialized
                robot_base_pos = self.get_calibrated_robot_base_pos()
                q = list(arm.get_joint_positions()) if hasattr(arm, 'get_joint_positions') else []
                is_cam_step = (self.step_counter % self.cam_decimation == 0)
                if is_cam_step and self.server_connected and self.initial_scan_done:
                    cam_data = self.capture_camera_data()
                    if cam_data:
                        blocking_send = (self.sync_mode == "stepped")
                        sim_ts = (self.step_counter * self.ctrl_period) if self.sync_mode == "stepped" else time.time()
                        self.client.push_frame_observation(
                            timestep=self.step_counter,
                            camera_dict=cam_data,
                            blocking=blocking_send,
                            robot_base_pos=robot_base_pos,
                            reachability_radius=self.reachability_radius,
                            joint_positions=q,
                            timestamp=sim_ts
                        )

                # 2. Read Robot State & Scene Target (if defined in task)
                try:
                    if not q and hasattr(arm, 'get_joint_positions'):
                        q = arm.get_joint_positions()
                    dq = arm.get_joint_velocities()
                    ee_pose = arm.get_tip().get_pose().tolist()
                    gripper_open = float(gripper.get_open_amount()[0])
                except Exception:
                    if not self.running:
                        break
                    continue

                target_pose = []
                target_available = False
                task_inst = getattr(self.task, '_task', self.task)
                if hasattr(task_inst, 'get_target_ee_pose'):
                    try:
                        target_pose = list(task_inst.get_target_ee_pose())
                        target_available = True
                    except Exception:
                        target_available = False
                elif hasattr(task_inst, 'target'):
                    try:
                        target_obj = task_inst.target
                        pos = list(target_obj.get_position())
                        if hasattr(task_inst, 'target_ee_orientation') and task_inst.target_ee_orientation is not None:
                            target_pose = pos + list(task_inst.target_ee_orientation)
                        else:
                            # Position-only goal if no EE orientation specified by task
                            target_pose = pos
                        target_available = True
                    except Exception:
                        target_available = False

                # 3. Request Control Action from DREMA MPC Controller
                if self.server_connected and self.task_active:
                    action = self.client.request_action(
                        timestep=self.step_counter,
                        joint_positions=q,
                        joint_velocities=dq,
                        ee_pose=ee_pose,
                        gripper_open=gripper_open,
                        task_active=self.task_active,
                        target_pose=target_pose,
                        target_available=target_available,
                        robot_base_pos=robot_base_pos,
                        reachability_radius=self.reachability_radius,
                        timeout=0.05 if self.sync_mode == "realtime" else 2.0
                    )
                    # 4. Actuate Robot
                    if not action.safety_stop and len(action.joint_velocities) == len(q):
                        arm.set_joint_target_velocities(action.joint_velocities)
                    else:
                        arm.set_joint_target_velocities([0.0] * len(q))

                    self.active_actions_count += 1
                    if self.active_actions_count <= 3 or (self.log_interval_actions > 0 and self.active_actions_count % self.log_interval_actions == 0):
                        v_max = max(abs(v) for v in action.joint_velocities) if action.joint_velocities else 0.0
                        d_tgt_str = ""
                        if target_available and len(target_pose) >= 3 and len(ee_pose) >= 3:
                            d_tgt = float(np.linalg.norm(np.array(ee_pose[:3]) - np.array(target_pose[:3])))
                            d_tgt_str = f" | d_tgt: {d_tgt:.3f}m"
                        print(
                            f"[COPPELIA ENVIRONMENT] Step #{self.step_counter:04d} (Act #{self.active_actions_count:04d}) | "
                            f"Stop: {action.safety_stop} | v_max: {v_max:.3f} rad/s{d_tgt_str} | Status: '{action.status_message}'"
                        )
                        q_str = "[" + ", ".join(f"{val:+.3f}" for val in q) + "]" if q else "[]"
                        qd_str = "[" + ", ".join(f"{val:+.3f}" for val in action.joint_velocities) + "]" if action.joint_velocities else "[]"
                        print(f"   ↳ q_pos  (J1..J7) [rad]:   {q_str}")
                        print(f"   ↳ qd_cmd (J1..J7) [rad/s]: {qd_str}")
                        if q and len(q) == 7:
                            q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
                            q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
                            near_lims = []
                            for j_i, val in enumerate(q):
                                d_lo = val - q_min[j_i]
                                d_hi = q_max[j_i] - val
                                if d_lo < 0.10:
                                    near_lims.append(f"J{j_i+1} near MIN ({d_lo:.3f} rad)")
                                elif d_hi < 0.10:
                                    near_lims.append(f"J{j_i+1} near MAX ({d_hi:.3f} rad)")
                            if near_lims:
                                print(f"   ↳ [LIMIT WARNING] {', '.join(near_lims)}")
                else:
                    # Hold position: zero target velocities
                    arm.set_joint_target_velocities([0.0] * len(q))

                # 5. Advance Task and CoppeliaSim Physics Step
                self.task._task.step()  # Moves oscillating tunnel obstacle
                self.env._pyrep.step()

                # 6. Check Task Success Condition (only when robot is actively executing task)
                if self.task_active:
                    success, terminate = self.task._task.success()
                    if success:
                        print(f"\n★ TASK SUCCESS ACHIEVED at step {self.step_counter}! Target touched cleanly.\n")
                        self.task_active = False

                # 7. Synchronization timing
                elapsed = time.time() - loop_start
                if self.sync_mode == "realtime":
                    sleep_time = max(0.0, self.ctrl_period - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[CoppeliaClient] Interrupted by user (Ctrl+C). Stopping simulation...")
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        try:
            self.client.close()
        except Exception:
            pass
        if hasattr(self, 'env'):
            try:
                self.env.shutdown()
            except Exception:
                pass
        print("✓ CoppeliaSim Client cleanly stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="CoppeliaSim / RLBench Closed-Loop Client for DREMA")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="DREMA gRPC server address (default: localhost:50051)")
    parser.add_argument("--task", type=str, default="dynamic_drema_test_1", help="Task name (default: dynamic_drema_test_1)")
    parser.add_argument("--sync_mode", type=str, choices=["stepped", "realtime"], default="realtime", help="Simulation sync mode: 'stepped' or 'realtime' (default: realtime)")
    parser.add_argument("--cam_fps", type=float, default=10.0, help="Camera sensor capture and streaming frequency in Hz (default: 10)")
    parser.add_argument("--ctrl_fps", type=float, default=50.0, help="Robot joint velocity control frequency in Hz (default: 50)")
    parser.add_argument("--reachability_radius", type=float, default=0.95, help="Robot maximum reachable radius in meters (default: 0.95)")
    parser.add_argument("--scan_resolution", type=int, nargs=2, default=[1280, 720], metavar=("WIDTH", "HEIGHT"),
                        help="Orbital scan camera resolution [width, height] (default: 1280 720)")
    parser.add_argument("--scan_steps", type=int, default=50, help="Number of orbital rotation steps at t=0 (default: 50)")
    parser.add_argument("--scan_chunk_size", type=int, default=2, help="Number of views per gRPC transmission chunk (default: 2)")
    parser.add_argument("--scan_chunk_timeout", type=float, default=180.0, help="Timeout in seconds per chunk transmission (default: 180.0)")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode (no GUI window)")
    parser.add_argument("--ping_timeout", type=float, default=1.5, help="gRPC ping timeout in seconds (default: 1.5)")
    parser.add_argument("--ping_retries", type=int, default=3, help="Consecutive failed pings before holding (default: 3)")
    parser.add_argument("--force_scan", action="store_true", default=False, help="Force orbital scan even if DREMA server has cached scene ready")
    parser.add_argument("--cam_resolution", type=int, nargs=2, default=[256, 256], metavar=("WIDTH", "HEIGHT"),
                        help="Streaming camera resolution [width, height] (default: 256 256)")
    parser.add_argument("--log_interval_actions", type=int, default=10,
                        help="Print control action telemetry every N steps (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = CoppeliaSimulationClient(
        server_address=args.server_address,
        task_name=args.task,
        sync_mode=args.sync_mode,
        headless=args.headless,
        cam_fps=args.cam_fps,
        ctrl_fps=args.ctrl_fps,
        reachability_radius=args.reachability_radius,
        scan_resolution=args.scan_resolution,
        scan_steps=args.scan_steps,
        scan_chunk_size=args.scan_chunk_size,
        scan_chunk_timeout=args.scan_chunk_timeout,
        ping_timeout=args.ping_timeout,
        ping_max_retries=args.ping_retries,
        force_scan=args.force_scan,
        cam_resolution=args.cam_resolution,
        log_interval_actions=args.log_interval_actions
    )
    client.run()
