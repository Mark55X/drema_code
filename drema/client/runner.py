#!/usr/bin/env python
"""
Closed-loop runner shared by every environment (CoppeliaSim, laboratory robot).

- Streams multi-camera RGB-D frames to DREMA (f_cam) once the initial scene scan is on the server.
- Queries the MPC controller every loop iteration (f_ctrl) with the robot state and the active keyframe, and applies
  the returned joint velocities through the environment's actuator.
- Interactive CLI: 'start' engages the task, 'stop'/'pause' halts, 'reset' restarts, 'quit' exits; object roles.
- --sync_mode stepped (lockstep, simulation only) or realtime (wall-clock continuous).
"""

import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..communication.grpc_client import DremaGrpcClient
from ..config import load_config
from ..roles.providers import ConfigRoleProvider, HumanRoleProvider, RoleManager, VLMSafetyProvider
from ..roles.set_of_marks import draw_marks
from ..roles.vlm import SceneObject, build_vlm_backend
from .environment import Environment
from .keyframes import KeyframeSequencer


class ClosedLoopRunner:

    def __init__(
        self,
        env: Environment,
        server_address: str = "localhost:50051",
        sync_mode: str = "realtime",
        cam_fps: float = 10.0,
        ctrl_fps: Optional[float] = None,
        scan_steps: int = 50,
        scan_chunk_size: int = 2,
        scan_chunk_timeout: float = 180.0,
        ping_timeout: float = 1.5,
        ping_max_retries: int = 3,
        force_scan: bool = False,
        reset_timeout: float = 30.0,
        log_interval_actions: int = 10,
        max_action_age: float = 0.20,
        roles_config: Optional[str] = "configs/roles_env.yaml",
        keyframes: str = "auto",
        policy: Optional[str] = None,
        settle_gripper: bool = True,
        settle_approach: bool = True
    ):
        """
        :param roles_config: Environment-side object roles (providers on/off, VLM endpoint); None disables them.
        :param keyframes: Keyframe source, one of the environment's keyframe_modes or 'auto'.
        :param policy: External keyframe policy 'module:function' (e.g. a PerAct wrapper), see PolicyKeyframeSource.
        :param settle_gripper: Actuate the gripper only once the arm has settled on the keypose.
        :param settle_approach: Approach poses of a grasp / release must be reached precisely, with the arm settled.
        """
        self.env = env
        self.tag = env.client_tag
        self.log_tag = env.log_tag
        self.server_address = server_address
        self.keyframe_mode = keyframes
        self.policy_spec = policy
        self.settle_gripper = bool(settle_gripper)
        self.settle_approach = bool(settle_approach)
        self.sequencer: Optional[KeyframeSequencer] = None
        self._init_roles(roles_config)
        self.sync_mode = sync_mode.lower()
        if self.sync_mode not in env.sync_modes:
            raise ValueError(f"Sync mode '{self.sync_mode}' not supported by this environment {env.sync_modes}")
        self.cam_fps = cam_fps
        self.ctrl_fps = ctrl_fps
        self.scan_steps = int(scan_steps)
        self.scan_chunk_size = int(scan_chunk_size)
        self.scan_chunk_timeout = float(scan_chunk_timeout)
        self.ping_timeout = float(ping_timeout)
        self.ping_max_retries = int(ping_max_retries)
        self.force_scan = bool(force_scan)
        self.reset_timeout = float(reset_timeout)
        self.log_interval_actions = max(0, int(log_interval_actions))

        # Interactive state
        self.task_active = False
        self.running = True
        self.step_counter = 0
        self.active_actions_count = 0
        self.keyframe_label = ""
        self._reset_requested = False
        self.server_connected = False
        self.initial_scan_done = False
        self._last_ping_time = 0.0
        self._failed_pings = 0
        self._last_read_warning = 0.0

        print(f"[{self.tag}] Connecting to DREMA suite at {self.server_address}...")
        self.client = DremaGrpcClient(target_address=self.server_address, max_action_age_s=max_action_age)
        is_alive, status = self.client.ping_status(timeout=self.ping_timeout)
        self.server_connected = is_alive
        if self.server_connected:
            print(f"✓ Connected to DREMA Dynamic Inference Suite! (Status: {status})")
            if status == "SCAN_READY":
                print("✓ DREMA server already has scene initialized (from disk cache or previous scan)! Ready for dynamic streaming.")
                self.initial_scan_done = True
        else:
            print(f"[Notice] DREMA Suite server not responding yet at {self.server_address}. The environment will run "
                  f"and connect automatically as soon as it goes online.")

        # Non-blocking camera streaming worker
        self.client.start_streaming()

        env.launch()
        self._prepare_keyframes()
        env.ready()

        if self.ctrl_fps is None:
            self.ctrl_fps = 1.0 / env.dt
        self.ctrl_period = 1.0 / self.ctrl_fps
        # cam_fps is an environment-time rate, consistent with the frame timestamps.
        self.cam_decimation = max(1, int(round(1.0 / (env.dt * self.cam_fps))))
        print(f"[{self.tag}] Environment timestep: {env.dt * 1000.0:.1f}ms | Control loop: {self.ctrl_fps:.1f}Hz | "
              f"Camera every {self.cam_decimation} steps ({1.0 / (self.cam_decimation * env.dt):.1f}Hz environment time)")
        if self.sync_mode == "realtime" and abs(self.ctrl_period - env.dt) > 1e-6:
            print(f"[{self.tag}] [Warning] realtime mode with ctrl period {self.ctrl_period * 1000.0:.1f}ms != environment "
                  f"timestep {env.dt * 1000.0:.1f}ms: the environment runs at {env.dt / self.ctrl_period:.2f}x wall-clock speed.")
        env.set_stream_rate(1.0 / (self.cam_decimation * env.dt))

        if self.force_scan:
            self.initial_scan_done = False
            if self.server_connected:
                self._reset_server_episode()

        self.cli_thread = threading.Thread(target=self._cli_listener, daemon=True)
        self.cli_thread.start()

    def _log(self, message: str) -> None:
        print(f"[{self.log_tag}] >>> {message}")

    def _reset_server_episode(self) -> None:
        """A server kept running across episodes still holds the previous scene: clear it before scanning."""
        ok = self.client.reset_episode(episode_index=0, task_name=self.env.task_name, timeout=self.reset_timeout)
        print(f"[{self.tag}] DREMA server episode reset before the scan: {'done' if ok else 'FAILED'}")

    def perform_initial_scan(self) -> bool:
        """Captures the initial scene views from the environment and sends them to DREMA in chunks."""
        print("\n=======================================================")
        print(f"[{self.tag}] Performing the initial scene scan...")
        print("=======================================================\n")
        try:
            scan = self.env.capture_scan(self.scan_steps)
            semantic_labels = dict(self.env.semantic_labels)
            print(f"[{self.tag} Scan] {len(scan.views)} views, {len(semantic_labels)} semantic labels.")
            res = self.client.push_initial_scan_batch(
                scan.views,
                semantic_labels=semantic_labels,
                robot_base_pos=self.env.robot_base_position(),
                reachability_radius=self.env.reachability_radius,
                joint_positions=scan.joint_positions,
                chunk_size=self.scan_chunk_size,
                chunk_timeout_s=self.scan_chunk_timeout,
                label_categories=self.env.label_categories,
                object_physics=getattr(self.env, 'object_physics', None) or {}
            )
        except Exception as e:
            print(f"[{self.tag} Error] Initial scan failed: {e}")
            return False

        if res and res.initial_scan_ready:
            print(f"\n✓ [{self.tag}] Initial scan complete! All {len(scan.views)} views sent. DREMA scene populated.")
            self.initial_scan_done = True
            self._on_scene_ready()
            return True
        print(f"\n❌ [{self.tag} Error] DREMA server failed to acknowledge initial scan (status: {res})!")
        print("   Dynamic tracking cannot proceed until the scene is initialized. Type 'scan' or 'reset' to retry.")
        self.initial_scan_done = False
        return False

    def _prepare_keyframes(self) -> None:
        """Builds the keyframe sequencer for the current episode (a demo source re-records its demo each episode)."""
        source = self.env.keyframe_source(self.keyframe_mode, self.policy_spec)
        if source is None:
            self.sequencer = None
            return
        self.sequencer = KeyframeSequencer(source, log=self._log,
                                           settle_on_gripper_change=self.settle_gripper,
                                           settle_on_precise_approach=self.settle_approach)

    # -------------------------------------------------------------------------
    # Object roles (environment side)
    # -------------------------------------------------------------------------

    def _init_roles(self, roles_config: Optional[str]) -> None:
        self.role_manager = RoleManager()
        self._role_requests: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.config_roles = self.vlm_safety = self.human_roles = None
        self.roles_cfg: Dict[str, Any] = {}
        self._last_vlm_safety = 0.0
        if not roles_config:
            return
        cfg = load_config(roles_config).get("roles", {}) if os.path.exists(roles_config) else {}
        if not cfg:
            print(f"[{self.tag}] [Notice] Roles config '{roles_config}' not found: environment roles disabled.")
            return
        if not bool(cfg.get("enabled", True)):
            return
        self.roles_cfg = cfg
        log = lambda m: print(f"[ROLES] {m}")
        vlm_cfg = cfg.get("vlm", {}) or {}
        human_cfg = cfg.get("human", {}) or {}
        vlm_on = bool(vlm_cfg.get("enabled", False))
        backend = build_vlm_backend(vlm_cfg) if vlm_on else None
        if bool(cfg.get("config_provider", {}).get("enabled", True)):
            self.config_roles = ConfigRoleProvider(cfg.get("config_provider", {}).get("rules", []) or [],
                                                   self.role_manager, log)
        if vlm_on and bool(cfg.get("vlm_safety", {}).get("enabled", False)):
            self.vlm_safety = VLMSafetyProvider(backend, self.role_manager, log)
        if bool(human_cfg.get("enabled", True)):
            self.human_roles = HumanRoleProvider(backend if bool(human_cfg.get("use_vlm", True)) else None,
                                                 self.role_manager, float(human_cfg.get("min_confidence", 0.7)),
                                                 str(human_cfg.get("confirm_relax", "uncertain")), log)
        print(f"[{self.tag}] Object roles: config={'on' if self.config_roles else 'off'}, "
              f"vlm_safety={'on' if self.vlm_safety else 'off'}, human={'on' if self.human_roles else 'off'} "
              f"(VLM backend: {vlm_cfg.get('backend', 'keyword') if backend else 'none'})")

    def _scene_objects(self) -> List[SceneObject]:
        """Objects offered to the VLM: those the server tracks when known, else the non-excluded object labels."""
        labels = self.env.semantic_labels
        names = {h: n for n, h in labels.items()}
        if self.role_manager.server_status:
            return [SceneObject(h, names.get(h, f"object_{h}")) for h in sorted(self.role_manager.server_status)]
        exclude = [k.lower() for k in self.roles_cfg.get("exclude_keywords", [])]
        return [SceneObject(h, n) for n, h in sorted(labels.items(), key=lambda kv: kv[1])
                if self.env.label_categories.get(h, 'object') == 'object' and not any(k in n.lower() for k in exclude)]

    def _vlm_image(self, objects: List[SceneObject]) -> Optional[np.ndarray]:
        cam = str(self.roles_cfg.get("vlm", {}).get("camera", "front"))
        data = self.env.capture_cameras().get(cam)
        if data is None:
            return None
        return draw_marks(data['rgb'], data.get('mask'), [o.object_id for o in objects])

    def _on_scene_ready(self) -> None:
        objects = self._scene_objects()
        if self.config_roles is not None:
            self.config_roles.apply(objects)
        if self.vlm_safety is not None and bool(self.roles_cfg.get("vlm_safety", {}).get("on_scan", True)):
            self._role_requests.put(("safety", None))

    def _process_role_requests(self) -> None:
        """Runs on the loop thread: camera captures must not be issued from the CLI thread."""
        period = float(self.roles_cfg.get("vlm_safety", {}).get("period_s", 0) or 0)
        if self.vlm_safety is not None and self.task_active and period > 0 and time.time() - self._last_vlm_safety > period:
            self._role_requests.put(("safety", None))
        while True:
            try:
                kind, payload = self._role_requests.get_nowait()
            except queue.Empty:
                return
            objects = self._scene_objects()
            image = self._vlm_image(objects) if objects else None
            if kind == "safety" and self.vlm_safety is not None:
                self._last_vlm_safety = time.time()
                self.vlm_safety.request(objects, image, self.env.task_description)
            elif kind == "say" and self.human_roles is not None:
                self.human_roles.submit_message(payload, objects, image, self.env.task_description)

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    def _cli_listener(self):
        time.sleep(0.5)
        print("\n=======================================================")
        print(f"  {self.tag} Interactive Console Ready!")
        print("  Commands:                                           ")
        print("    'start' (or press Enter) -> Begin task execution  ")
        print("    'stop'  (or 'pause')     -> Halt robot motion     ")
        print("    'scan'                   -> Scan the scene again  ")
        print("    'reset'                  -> Reset episode         ")
        print("    'quit'  (or 'exit')      -> Exit                  ")
        print("  Object roles:                                       ")
        print("    'say <text>'             -> Tell the robot about an object (VLM)")
        print("    'yes' / 'no'             -> Confirm / cancel a pending permission")
        print("    'role <id> <touch|avoid|fragile|clear>'           ")
        print("    'roles'                  -> Show object roles     ")
        print("    'vlm'                    -> Run the VLM safety check now")
        print("=======================================================\n")

        while self.running:
            try:
                raw = sys.stdin.readline()
                if not raw:
                    break
                raw = raw.strip()
                cmd = raw.lower()
                if self._handle_role_command(raw, cmd):
                    continue

                if cmd in ['start', 'run']:
                    self.task_active = True
                    print(f"\n[{self.log_tag}] >>> TASK STARTED! Robot closed-loop control engaged (f_ctrl={self.ctrl_fps}Hz).\n")
                    if not self.server_connected:
                        print(f"[{self.log_tag}] [Notice] Waiting for DREMA suite connection before actuating robot...\n")
                elif cmd in ['scan', 's']:
                    print(f"\n[{self.log_tag}] >>> Initiating initial scene scan...")
                    self.perform_initial_scan()
                elif cmd in ['stop', 'pause', 'halt']:
                    self.task_active = False
                    print(f"\n[{self.log_tag}] >>> TASK PAUSED! Robot holding position (streaming continues).\n")
                elif cmd in ['reset', 'r']:
                    self.task_active = False
                    self._reset_requested = True
                    print(f"\n[{self.log_tag}] >>> Reset requested. Will reset cleanly on next loop tick.\n")
                elif cmd in ['quit', 'exit', 'q']:
                    print(f"\n[{self.log_tag}] >>> Quitting...")
                    self.running = False
                    break
                else:
                    print(f"[{self.log_tag}] Unknown command '{cmd}'. Type 'start', 'stop', 'reset', or 'quit'.")
            except Exception:
                break

    def _handle_role_command(self, raw: str, cmd: str) -> bool:
        """Console commands about object roles; returns True when the line was one of them."""
        names = {h: n for n, h in self.env.semantic_labels.items()}
        word = cmd.split(" ", 1)[0]
        if word == "say":
            text = raw.split(" ", 1)[1] if " " in raw else ""
            if self.human_roles is None:
                print("[ROLES] Human role provider disabled (configs/roles_env.yaml: roles.human.enabled)")
            elif text:
                self._role_requests.put(("say", text))
            return True
        if cmd in ("yes", "y", "no", "n"):
            if self.human_roles is None:
                return False
            self.human_roles.confirm(cmd in ("yes", "y"))
            return True
        if word == "role":
            parts = cmd.split()
            if self.human_roles is None or len(parts) != 3 or not parts[1].lstrip("-").isdigit() \
                    or parts[2] not in ("touch", "avoid", "fragile", "clear"):
                print("[ROLES] Usage: role <id> <touch|avoid|fragile|clear>  (ids: 'roles')")
            else:
                # An exact id typed by the operator needs no grounding, hence no confirmation.
                self.human_roles.apply(int(parts[1]), parts[2], names, reason="console")
            return True
        if cmd == "roles":
            print("[ROLES] Scene objects offered to the VLM: "
                  + ", ".join(f"{o.object_id}={o.name}" for o in self._scene_objects()))
            print(self.role_manager.describe(names))
            return True
        if cmd == "vlm":
            if self.vlm_safety is None:
                print("[ROLES] VLM safety provider disabled (configs/roles_env.yaml: roles.vlm_safety.enabled)")
            else:
                self._role_requests.put(("safety", None))
            return True
        return False

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def _reset_episode(self) -> None:
        env = self.env
        self.task_active = False
        self.step_counter = 0
        self.active_actions_count = 0
        self.initial_scan_done = False
        if self.server_connected:
            try:
                self.client.reset_episode(episode_index=0, task_name=env.task_name, timeout=self.reset_timeout)
            except Exception:
                pass
        env.reset()
        self.role_manager.reset()
        self._prepare_keyframes()
        env.gripper.reset()
        env.actuator.reset()
        print(f"\n[{self.log_tag}] >>> EPISODE RESET COMPLETE! Re-scanning initial scene...\n")
        if self.server_connected:
            self.perform_initial_scan()
        print(f"\n[{self.log_tag}] >>> Ready. Press 'start' to resume dynamic task.\n")

    def _check_connection(self, now: float) -> None:
        """Periodic non-blocking connection check to the DREMA suite."""
        if now - self._last_ping_time <= 1.5:
            return
        self._last_ping_time = now
        is_alive, status = self.client.ping_status(timeout=self.ping_timeout)
        if is_alive:
            self._failed_pings = 0
            if not self.server_connected:
                self.server_connected = True
                print(f"\n✓ [{self.tag}] Connected to DREMA Dynamic Inference Suite! (Status: {status})\n")
                if self.force_scan and not self.initial_scan_done:
                    self._reset_server_episode()
                    self.perform_initial_scan()
                elif status == "SCAN_READY":
                    print(f"✓ [{self.tag}] DREMA server already has scene initialized! Ready for dynamic streaming.\n")
                    self.initial_scan_done = True
                    self._on_scene_ready()
                elif not self.initial_scan_done:
                    self.perform_initial_scan()
        else:
            self._failed_pings += 1
            if self._failed_pings >= self.ping_max_retries and self.server_connected:
                self.server_connected = False
                print(f"\n[Notice] [{self.tag}] DREMA Suite disconnected (missed {self.ping_max_retries} consecutive "
                      f"pings). Holding position.\n")

    def _keyframe_target(self, ee_pose: List[float]) -> Dict[str, Any]:
        """Active keyframe (or the task's fallback target) as the RobotState target fields."""
        target = {'target_pose': [], 'target_available': False, 'approach_funnel': None, 'gripper_command': None,
                  'ignore_collisions': None, 'target_in_tip_frame': False}
        if self.sequencer is not None:
            try:
                kf = self.sequencer.current(self.env.observe)
                if kf is not None:
                    if self.task_active:
                        self.sequencer.step(ee_pose, self.env.gripper)
                        kf = self.sequencer.current(self.env.observe)
                    target.update(target_pose=list(kf.pose), approach_funnel=kf.approach_funnel,
                                  gripper_command=kf.gripper_open, ignore_collisions=kf.contact_flag(),
                                  target_in_tip_frame=kf.frame == "tip", target_available=True)
                    self.keyframe_label = (f"{kf.name} (#{self.sequencer.index + 1}"
                                           f"{', done' if self.sequencer.finished else ''}"
                                           f"{', gripper' if self.sequencer.phase == 'gripper' else ''})")
            except Exception as e:
                print(f"[{self.tag}] [Warning] Keyframe sequencing failed: {e}")
                target['target_available'] = False
            return target
        try:
            pose, available = self.env.fallback_target()
        except Exception:
            pose, available = [], False
        target.update(target_pose=list(pose), target_available=bool(available))
        return target

    def _log_action(self, action: Any, q: List[float], ee_pose: List[float], target: Dict[str, Any]) -> None:
        v_max = max(abs(v) for v in action.joint_velocities) if action.joint_velocities else 0.0
        d_tgt_str = ""
        target_pose = target['target_pose']
        if target['target_available'] and len(target_pose) >= 3 and len(ee_pose) >= 3:
            d_tgt = float(np.linalg.norm(np.array(ee_pose[:3]) - np.array(target_pose[:3])))
            d_tgt_str = f" | d_tgt: {d_tgt:.3f}m"
        kf_str = f" | kf: {self.keyframe_label}" if self.keyframe_label else ""
        print(
            f"[{self.log_tag}] Step #{self.step_counter:04d} (Act #{self.active_actions_count:04d}) | "
            f"Stop: {action.safety_stop} | v_max: {v_max:.3f} rad/s{d_tgt_str}{kf_str} | Status: '{action.status_message}'"
        )
        q_str = "[" + ", ".join(f"{val:+.3f}" for val in q) + "]" if q else "[]"
        qd_str = "[" + ", ".join(f"{val:+.3f}" for val in action.joint_velocities) + "]" if action.joint_velocities else "[]"
        print(f"   ↳ q_pos  (J1..J7) [rad]:   {q_str}")
        print(f"   ↳ qd_cmd (J1..J7) [rad/s]: {qd_str}")
        if q and len(q) == 7:
            q_min, q_max = self.env.joint_limits()
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

    def run(self):
        """Main loop balancing sensing and control rates."""
        env = self.env
        print(f"[{self.tag}] Loop started (Sync Mode: {self.sync_mode}).")
        print(f"[{self.tag}] Multi-camera streaming active immediately. Robot is IDLE until 'start' command.")
        env.describe()

        if self.server_connected and not self.initial_scan_done:
            self.perform_initial_scan()
        elif self.initial_scan_done:
            self._on_scene_ready()

        try:
            while self.running:
                loop_start = time.time()

                # Asynchronous reset request from the CLI, handled on the loop thread
                if self._reset_requested:
                    self._reset_requested = False
                    self._reset_episode()
                    continue

                self._check_connection(loop_start)
                self.step_counter += 1

                # 1. Perception step (f_cam): frames are pushed only once the scene is initialized
                robot_base_pos = env.robot_base_position()
                try:
                    reading = env.read_robot()
                except Exception as e:
                    if not self.running:
                        break
                    if time.time() - self._last_read_warning > 2.0:
                        self._last_read_warning = time.time()
                        print(f"[{self.tag}] [Warning] Robot state unavailable, holding: {e}")
                    env.actuator.hold()
                    if self.sync_mode == "realtime":
                        time.sleep(self.ctrl_period)
                    continue
                q = list(reading.joint_positions)
                is_cam_step = (self.step_counter % self.cam_decimation == 0)
                if is_cam_step and self.server_connected and self.initial_scan_done:
                    cam_data = env.capture_cameras()
                    if cam_data:
                        self.client.push_frame_observation(
                            timestep=self.step_counter,
                            camera_dict=cam_data,
                            blocking=(self.sync_mode == "stepped"),
                            robot_base_pos=robot_base_pos,
                            reachability_radius=env.reachability_radius,
                            joint_positions=q,
                            timestamp=env.frames_timestamp(cam_data)
                        )

                # 2. Robot state and active target
                dq = list(reading.joint_velocities)
                ee_pose = list(reading.ee_pose)
                self._process_role_requests()
                target = self._keyframe_target(ee_pose)

                # 3. Control action from the DREMA MPC controller
                if self.server_connected and self.task_active:
                    roles_revision, role_assignments = self.role_manager.snapshot()
                    action = self.client.request_action(
                        timestep=self.step_counter,
                        joint_positions=q,
                        joint_velocities=dq,
                        ee_pose=ee_pose,
                        gripper_open=reading.gripper_open,
                        task_active=self.task_active,
                        target_pose=target['target_pose'],
                        target_available=target['target_available'],
                        robot_base_pos=robot_base_pos,
                        reachability_radius=env.reachability_radius,
                        timeout=(2.0 if self.sync_mode == "stepped" else 0.5 * self.ctrl_period),
                        blocking=(self.sync_mode == "stepped"),
                        approach_funnel=target['approach_funnel'],
                        gripper_command=target['gripper_command'],
                        ignore_collisions=target['ignore_collisions'],
                        keyframe_index=self.sequencer.index if self.sequencer is not None else 0,
                        target_in_tip_frame=target['target_in_tip_frame'],
                        yaw_symmetric=target['target_in_tip_frame'],
                        roles_revision=roles_revision,
                        role_assignments=role_assignments,
                        gripper_actuated=env.gripper.target if env.gripper is not None else None,
                        gripper_moving=env.gripper.busy if env.gripper is not None else None
                    )
                    self.role_manager.update_server_status(action.object_roles)
                    # 4. Actuate the robot
                    if not action.safety_stop and len(action.joint_velocities) == len(q):
                        env.actuator.apply(action.joint_velocities)
                    else:
                        env.actuator.hold()

                    self.active_actions_count += 1
                    if self.active_actions_count <= 3 or (self.log_interval_actions > 0
                                                          and self.active_actions_count % self.log_interval_actions == 0):
                        self._log_action(action, q, ee_pose, target)
                else:
                    env.actuator.hold()

                # 5. Advance gripper, task and environment by one step
                env.advance(self.step_counter, self.task_active)

                # 6. Task success (only while the robot is executing the task)
                if self.task_active and env.check_success():
                    print(f"\n★ TASK SUCCESS ACHIEVED at step {self.step_counter}! ({env.task_name})\n")
                    self.task_active = False

                # 7. Synchronization timing
                elapsed = time.time() - loop_start
                if self.sync_mode == "realtime":
                    sleep_time = max(0.0, self.ctrl_period - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n[{self.tag}] Interrupted by user (Ctrl+C). Stopping...")
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        try:
            self.env.actuator.hold()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self.env.shutdown()
        except Exception:
            pass
        print(f"✓ {self.tag} cleanly stopped.")
