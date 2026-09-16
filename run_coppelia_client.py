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
        ctrl_fps: float = 50.0
    ):
        self.server_address = server_address
        self.task_name = task_name
        self.sync_mode = sync_mode.lower()
        self.headless = headless
        self.cam_fps = cam_fps
        self.ctrl_fps = ctrl_fps

        self.cam_period = 1.0 / max(1.0, cam_fps)
        self.ctrl_period = 1.0 / max(1.0, ctrl_fps)
        self.cam_decimation = max(1, int(round(self.ctrl_fps / self.cam_fps)))

        # Interactive state
        self.task_active = False
        self.running = True
        self.step_counter = 0
        self._reset_requested = False
        self.server_connected = False
        self._last_ping_time = 0.0

        # Initialize gRPC Client
        print(f"[CoppeliaClient] Connecting to DREMA suite at {self.server_address}...")
        self.client = DremaGrpcClient(target_address=self.server_address)
        self.server_connected = self.client.ping(timeout=0.5)
        if self.server_connected:
            print(f"✓ Connected to DREMA Dynamic Inference Suite!")
        else:
            print(f"[Notice] DREMA Suite server not responding yet at {self.server_address}. Simulation will run and connect automatically as soon as it goes online.")

        # Start non-blocking camera streaming worker
        self.client.start_streaming()

        # Initialize RLBench Environment
        self._init_rlbench()

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
        resolutions = [128, 128]
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
            p = list(base_pose)
            p[0] += p_offsets[idx][0]
            p[1] += p_offsets[idx][1]
            p[2] += p_offsets[idx][2]
            c.set_pose(p)
            c.set_parent(placeholder)
            cams.append(c)

        rotate_angle = (2.0 * np.pi) / float(num_steps)
        all_scan_cameras = []

        try:
            for step_idx in range(num_steps):
                base_dummy.rotate([0, 0, rotate_angle])

                for c_idx, c in enumerate(cams):
                    # In CoppeliaSim, explicit_handling requires handle_explicit() before capture
                    c.handle_explicitly()
                    rgb_float = c.capture_rgb()
                    rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                    depth_m = c.capture_depth(in_meters=True).astype(np.float32)
                    ext = np.array(c.get_matrix(), dtype=np.float32).reshape((4, 4))
                    intrinsic = np.array(c.get_intrinsic_matrix(), dtype=np.float32)

                    cam_name = f'orbit_{step_idx}_{c_idx}'
                    all_scan_cameras.append((cam_name, rgb_uint8, depth_m, ext, intrinsic))

                if (step_idx + 1) % 10 == 0:
                    print(f"[CoppeliaClient Scan] Captured {(step_idx + 1) * 4}/{num_steps * 4} views...")

            # Extract semantic labels from CoppeliaSim scene shapes
            from pyrep.backend import sim
            from pyrep.objects.shape import Shape
            handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
            semantic_labels = {}
            filter_names = ["DefaultCamera", "ResizableFloor", "workspace", "Wall"]
            for h in handles:
                try:
                    name = Shape(h).get_name()
                    if any(f in name for f in filter_names):
                        continue
                    semantic_labels[name] = int(h)
                except Exception:
                    pass

            print(f"[CoppeliaClient Scan] Extracted {len(semantic_labels)} semantic labels from CoppeliaSim scene.")

            # Send ALL views in ONE single batch message
            res = self.client.push_initial_scan_batch(all_scan_cameras, semantic_labels)

            if res and res.initial_scan_ready:
                print(f"\n✓ [CoppeliaClient] 360° orbital scan complete! All {len(all_scan_cameras)} views sent in single batch. DREMA scene populated.")
                self.initial_scan_done = True
                return True
            else:
                print(f"[CoppeliaClient Warning] DREMA server acknowledged initial scan with status: {res}")
                self.initial_scan_done = True
                return True

        except Exception as e:
            print(f"[CoppeliaClient Error] Orbital scan failed: {e}")
            return False
        finally:
            for c in cams:
                try:
                    c.remove()
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
        obs_config.front_camera.image_size = (128, 128)

        obs_config.wrist_camera.rgb = True
        obs_config.wrist_camera.depth = True
        obs_config.wrist_camera.image_size = (128, 128)

        obs_config.overhead_camera.rgb = True
        obs_config.overhead_camera.depth = True
        obs_config.overhead_camera.image_size = (128, 128)

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

                if cmd in ['start', 'run', '']:
                    self.task_active = True
                    print(f"\n[CLI] >>> TASK STARTED! Robot closed-loop control engaged (f_ctrl={self.ctrl_fps}Hz).\n")
                elif cmd in ['stop', 'pause', 'halt']:
                    self.task_active = False
                    print("\n[CLI] >>> TASK PAUSED! Robot holding position (streaming continues).\n")
                elif cmd in ['reset', 'r']:
                    self.task_active = False
                    self._reset_requested = True
                    print("\n[CLI] >>> Reset requested. Will reset cleanly on next simulation tick.\n")
                elif cmd in ['quit', 'exit', 'q']:
                    print("\n[CLI] >>> Quitting simulation...")
                    self.running = False
                    break
                else:
                    print(f"[CLI] Unknown command '{cmd}'. Type 'start', 'stop', 'reset', or 'quit'.")
            except Exception:
                break

    def capture_camera_data(self):
        """Extracts RGB, Depth, Extrinsics, and Intrinsics from all vision sensors."""
        cam_dict = {}
        for name, cam in self.cameras.items():
            # RGB: uint8 [0, 255]
            rgb_float = cam.capture_rgb()
            rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

            # Depth: float32 in meters
            depth_m = cam.capture_depth(in_meters=True).astype(np.float32)

            # 4x4 Cam-to-world pose
            ext = np.array(cam.get_matrix(), dtype=np.float32).reshape((4, 4))

            # 3x3 Intrinsic matrix
            intrinsic = np.array(cam.get_intrinsic_matrix(), dtype=np.float32)

            cam_dict[name] = {
                'rgb': rgb_uint8,
                'depth': depth_m,
                'extrinsics': ext,
                'intrinsics': intrinsic
            }
        return cam_dict

    def run(self):
        """Main execution loop balancing sensing and control rates."""
        print(f"[CoppeliaClient] Simulation loop started (Sync Mode: {self.sync_mode}).")
        print("[CoppeliaClient] Multi-camera streaming active immediately. Robot is IDLE until 'start' command.")

        robot = self.task._robot
        arm = robot.arm
        gripper = robot.gripper

        # If server is already online at launch, perform initial scan immediately
        if self.server_connected and not self.initial_scan_done:
            self.perform_initial_scan()

        try:
            while self.running:
                loop_start = time.time()

                # Handle asynchronous reset request from CLI safely on the main thread
                if self._reset_requested:
                    self._reset_requested = False
                    self.task_active = False
                    self.step_counter = 0
                    self.initial_scan_done = False
                    if self.server_connected:
                        try:
                            self.client.reset_episode(episode_index=0, task_name=self.task_name, timeout=0.5)
                        except Exception:
                            pass
                    descriptions, self.current_obs = self.task.reset()
                    print("\n[CLI] >>> EPISODE RESET COMPLETE! Re-scanning initial scene...\n")
                    if self.server_connected:
                        self.perform_initial_scan()
                    print("\n[CLI] >>> Ready. Press 'start' to resume dynamic task.\n")
                    continue

                # Periodic non-blocking connection check to DREMA suite
                if loop_start - self._last_ping_time > 1.5:
                    self._last_ping_time = loop_start
                    is_alive = self.client.ping(timeout=0.1)
                    if is_alive and not self.server_connected:
                        self.server_connected = True
                        print("\n✓ [CoppeliaClient] Connected to DREMA Dynamic Inference Suite!\n")
                        if not self.initial_scan_done:
                            self.perform_initial_scan()
                    elif not is_alive and self.server_connected:
                        self.server_connected = False
                        print("\n[Notice] [CoppeliaClient] DREMA Suite disconnected. Holding position.\n")

                self.step_counter += 1

                # 1. Perception Step (f_cam ≈ 10 Hz): capture and push frames to gRPC queue if connected
                is_cam_step = (self.step_counter % self.cam_decimation == 0)
                if is_cam_step and self.server_connected:
                    cam_data = self.capture_camera_data()
                    blocking_send = (self.sync_mode == "stepped")
                    self.client.push_frame_observation(
                        timestep=self.step_counter,
                        camera_dict=cam_data,
                        blocking=blocking_send
                    )

                # 2. Read Robot State & Scene Target (if defined in task)
                q = arm.get_joint_positions()
                dq = arm.get_joint_velocities()
                ee_pose = arm.get_tip().get_pose().tolist()
                gripper_open = float(gripper.get_open_amount()[0])

                target_pose = []
                target_available = False
                if hasattr(self.task, '_task') and hasattr(self.task._task, 'target'):
                    try:
                        target_obj = self.task._task.target
                        if hasattr(target_obj, 'get_pose'):
                            target_pose = target_obj.get_pose().tolist()
                        elif hasattr(target_obj, 'get_position'):
                            target_pose = list(target_obj.get_position()) + [0.0, 0.0, 0.0, 1.0]
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
                        timeout=0.05 if self.sync_mode == "realtime" else 2.0
                    )
                    # 4. Actuate Robot
                    if not action.safety_stop and len(action.joint_velocities) == len(q):
                        arm.set_joint_target_velocities(action.joint_velocities)
                    else:
                        arm.set_joint_target_velocities([0.0] * len(q))
                else:
                    # Hold position: zero target velocities
                    arm.set_joint_target_velocities([0.0] * len(q))

                # 5. Advance Task and CoppeliaSim Physics Step
                self.task._task.step()  # Moves oscillating tunnel obstacle
                self.env._pyrep.step()

                # 6. Check Task Success Condition
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
            print("\n[CoppeliaClient] Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        self.client.close()
        if hasattr(self, 'env'):
            self.env.shutdown()
        print("✓ CoppeliaSim Client cleanly stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="CoppeliaSim / RLBench Closed-Loop Client for DREMA")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="DREMA gRPC server address (default: localhost:50051)")
    parser.add_argument("--task", type=str, default="dynamic_drema_test_1", help="Task name (default: dynamic_drema_test_1)")
    parser.add_argument("--sync_mode", type=str, choices=["stepped", "realtime"], default="realtime", help="Simulation sync mode: 'stepped' or 'realtime' (default: realtime)")
    parser.add_argument("--cam_fps", type=float, default=10.0, help="Camera sensor capture and streaming frequency in Hz (default: 10)")
    parser.add_argument("--ctrl_fps", type=float, default=50.0, help="Robot joint velocity control frequency in Hz (default: 50)")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode (no GUI window)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = CoppeliaSimulationClient(
        server_address=args.server_address,
        task_name=args.task,
        sync_mode=args.sync_mode,
        headless=args.headless,
        cam_fps=args.cam_fps,
        ctrl_fps=args.ctrl_fps
    )
    client.run()
