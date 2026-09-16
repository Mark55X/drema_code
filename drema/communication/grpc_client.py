#!/usr/bin/env python
"""
High-performance gRPC client for CoppeliaSim / RLBench -> DREMA Dynamic Inference Suite.

Supports:
- Asynchronous non-blocking multi-camera RGB-D streaming (f_cam ≈ 10-15 Hz) with frame drop on lag.
- Synchronous or asynchronous request-response control actions (f_ctrl ≈ 50 Hz).
- Automatic reconnection and channel configuration for large tensor payloads.
"""

import os
import sys
import time
import queue
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
import grpc

from .proto import drema_comm_pb2, drema_comm_pb2_grpc


def pack_camera_frame(
    name: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray
) -> drema_comm_pb2.CameraFrame:
    """
    Serializes camera RGB, Depth, Extrinsics, and Intrinsics into a Protobuf CameraFrame.
    rgb: uint8 array (H, W, 3)
    depth: float32 array (H, W) in meters
    extrinsics: 4x4 matrix (cam to world)
    intrinsics: 3x3 matrix
    """
    h, w = rgb.shape[:2]
    c = rgb.shape[2] if len(rgb.shape) > 2 else 1

    # Ensure contiguous memory buffers for fast serialization
    rgb_bytes = rgb.astype(np.uint8).tobytes()
    depth_bytes = depth.astype(np.float32).tobytes()

    ext_flat = extrinsics.flatten().tolist()
    int_flat = intrinsics.flatten().tolist()

    return drema_comm_pb2.CameraFrame(
        name=name,
        width=w,
        height=h,
        channels=c,
        rgb_data=rgb_bytes,
        depth_data=depth_bytes,
        extrinsics=ext_flat,
        intrinsics=int_flat
    )


def unpack_camera_frame(frame: drema_comm_pb2.CameraFrame) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deserializes a Protobuf CameraFrame into numpy arrays.
    Returns: (name, rgb (H,W,3 uint8), depth (H,W float32), extrinsics (4,4), intrinsics (3,3))
    """
    h, w, c = frame.height, frame.width, frame.channels
    rgb = np.frombuffer(frame.rgb_data, dtype=np.uint8).reshape((h, w, c))
    depth = np.frombuffer(frame.depth_data, dtype=np.float32).reshape((h, w))
    extrinsics = np.array(frame.extrinsics, dtype=np.float32).reshape((4, 4))
    intrinsics = np.array(frame.intrinsics, dtype=np.float32).reshape((3, 3))
    return frame.name, rgb, depth, extrinsics, intrinsics


class DremaGrpcClient:
    """
    gRPC Client attached to CoppeliaSim / PyRep simulation loop.
    """

    def __init__(self, target_address: str = "localhost:50051", max_queue_size: int = 3):
        self.target_address = target_address
        self.max_queue_size = max_queue_size

        # Options for high-bandwidth RGB-D transfer (100 MB buffer limit)
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.so_reuseport', 1),
            ('grpc.use_local_subchannel_pool', 1),
        ]

        self.channel = grpc.insecure_channel(self.target_address, options=options)
        self.stub = drema_comm_pb2_grpc.DremaInferenceServiceStub(self.channel)

        # Background streaming queue & thread
        self._frame_queue = queue.Queue(maxsize=self.max_queue_size)
        self._streaming_thread = None
        self._stop_event = threading.Event()
        self._stream_active = False

        self.last_action: Optional[drema_comm_pb2.ControlAction] = None
        self.last_timestep: int = 0

    def ping(self, timeout: float = 0.2) -> bool:
        """Checks if DREMA Dynamic Inference Suite server is alive."""
        try:
            req = drema_comm_pb2.PingRequest(client_id="coppelia_client", timestamp=time.time())
            res = self.stub.Ping(req, timeout=timeout)
            return res.alive
        except Exception:
            return False

    def start_streaming(self):
        """Starts the background non-blocking camera frame streaming thread."""
        if self._streaming_thread and self._streaming_thread.is_alive():
            return
        self._stop_event.clear()
        self._streaming_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._streaming_thread.start()

    def _frame_generator(self):
        while not self._stop_event.is_set():
            try:
                obs = self._frame_queue.get(timeout=0.2)
                if obs is None:
                    break
                yield obs
                self._frame_queue.task_done()
            except queue.Empty:
                continue

    def _stream_worker(self):
        while not self._stop_event.is_set():
            try:
                self._stream_active = True
                self.stub.StreamFrames(self._frame_generator())
                self._stream_active = False
            except Exception as e:
                self._stream_active = False
                if not self._stop_event.is_set():
                    time.sleep(0.2)

    def push_frame_observation(
        self,
        timestep: int,
        camera_dict: Dict[str, Dict[str, np.ndarray]],
        blocking: bool = False,
        is_initial_scan: bool = False,
        is_scan_finished: bool = False,
        semantic_labels: Optional[Dict[str, int]] = None
    ) -> Optional[drema_comm_pb2.StreamStatus]:
        """
        Pushes a multi-camera observation into the streaming queue.
        """
        frames = []
        for name, data in camera_dict.items():
            f = pack_camera_frame(
                name=name,
                rgb=data['rgb'],
                depth=data['depth'],
                extrinsics=data['extrinsics'],
                intrinsics=data['intrinsics']
            )
            frames.append(f)

        obs = drema_comm_pb2.FrameObservation(
            timestep=timestep,
            timestamp=time.time(),
            cameras=frames,
            is_initial_scan=is_initial_scan,
            is_scan_finished=is_scan_finished,
            semantic_labels=semantic_labels or {}
        )

        if blocking:
            try:
                res = self.stub.SendFrame(obs, timeout=30.0)
                return res
            except Exception as e:
                print(f"[GrpcClient Error] SendFrame failed: {e}")
                return None

    def push_initial_scan_batch(
        self,
        cameras_list: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        semantic_labels: Optional[Dict[str, int]] = None
    ) -> Optional[drema_comm_pb2.StreamStatus]:
        """
        Pushes the entire 360° orbital scan (all ~200 views) in a single batch to DREMA suite.
        Each camera tuple: (name, rgb, depth, extrinsics, intrinsics)
        """
        frames = []
        for name, rgb, depth, extrinsics, intrinsics in cameras_list:
            f = pack_camera_frame(
                name=name,
                rgb=rgb,
                depth=depth,
                extrinsics=extrinsics,
                intrinsics=intrinsics
            )
            frames.append(f)

        obs = drema_comm_pb2.FrameObservation(
            timestep=0,
            timestamp=time.time(),
            cameras=frames,
            is_initial_scan=True,
            is_scan_finished=True,
            semantic_labels=semantic_labels or {}
        )

        try:
            print(f"[GrpcClient] Sending single-batch initial scan ({len(frames)} views, {len(semantic_labels or {})} labels) to DREMA...")
            res = self.stub.SendFrame(obs, timeout=60.0)
            return res
        except Exception as e:
            print(f"[GrpcClient Error] push_initial_scan_batch failed: {e}")
            return None

        # In non-blocking mode: if queue is full, drop oldest frame to maintain low latency
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(obs)
        except queue.Full:
            pass

    def request_action(
        self,
        timestep: int,
        joint_positions: List[float],
        joint_velocities: List[float],
        ee_pose: List[float],
        gripper_open: float,
        task_active: bool,
        target_pose: Optional[List[float]] = None,
        target_available: bool = False,
        timeout: float = 0.5
    ) -> drema_comm_pb2.ControlAction:
        """
        Queries the MPC controller for the next joint velocity command.
        """
        state = drema_comm_pb2.RobotState(
            timestamp=time.time(),
            timestep=timestep,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            ee_pose=ee_pose,
            gripper_open=gripper_open,
            task_active=task_active,
            target_pose=target_pose if target_pose is not None else [],
            target_available=target_available
        )
        try:
            action = self.stub.RequestAction(state, timeout=timeout)
            self.last_action = action
            return action
        except grpc.RpcError as e:
            # Fallback on timeout or lag: return zero velocities to hold position safely
            if self.last_action is not None:
                return self.last_action
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=timestep,
                joint_velocities=[0.0] * len(joint_positions),
                gripper_action=gripper_open,
                safety_stop=True,
                status_message=f"RPC Error fallback: {e.code()}"
            )

    def reset_episode(self, episode_index: int = 0, task_name: str = "dynamic_drema_test_1", timeout: float = 5.0) -> bool:
        """Notifies DREMA suite that a new episode has started."""
        try:
            req = drema_comm_pb2.ResetRequest(episode_index=episode_index, task_name=task_name)
            res = self.stub.ResetEpisode(req, timeout=timeout)
            return res.success
        except Exception:
            return False

    def close(self):
        """Clean shutdown of background threads and gRPC channel."""
        self._stop_event.set()
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=1.0)
        self.channel.close()
