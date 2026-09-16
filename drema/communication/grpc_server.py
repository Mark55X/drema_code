#!/usr/bin/env python
"""
gRPC Server implementation for DREMA Dynamic Inference Suite.

Hosts the DremaInferenceService to receive continuous streaming camera frames
and respond to robot state control queries.
"""

import time
import threading
from concurrent import futures
from typing import Callable, Optional
import grpc

from .proto import drema_comm_pb2, drema_comm_pb2_grpc


class DremaInferenceServicer(drema_comm_pb2_grpc.DremaInferenceServiceServicer):
    """
    gRPC Servicer routing requests between the client and DREMA suite components.
    """

    def __init__(
        self,
        on_frame_callback: Optional[Callable[[drema_comm_pb2.FrameObservation], None]] = None,
        on_action_callback: Optional[Callable[[drema_comm_pb2.RobotState], drema_comm_pb2.ControlAction]] = None,
        on_reset_callback: Optional[Callable[[drema_comm_pb2.ResetRequest], bool]] = None
    ):
        self.on_frame_callback = on_frame_callback
        self.on_action_callback = on_action_callback
        self.on_reset_callback = on_reset_callback

        self.received_frames_count = 0
        self.last_timestep = 0
        self.lock = threading.Lock()

    def Ping(self, request: drema_comm_pb2.PingRequest, context) -> drema_comm_pb2.PingResponse:
        return drema_comm_pb2.PingResponse(
            alive=True,
            server_timestamp=time.time(),
            suite_status="DREMA Inference Suite Online"
        )

    def SendFrame(self, request: drema_comm_pb2.FrameObservation, context) -> drema_comm_pb2.StreamStatus:
        with self.lock:
            self.received_frames_count += 1
            self.last_timestep = request.timestep

        if self.on_frame_callback:
            try:
                res = self.on_frame_callback(request)
                if isinstance(res, drema_comm_pb2.StreamStatus):
                    return res
            except Exception as e:
                return drema_comm_pb2.StreamStatus(
                    success=False,
                    message=f"Error in on_frame_callback: {str(e)}",
                    received_timestep=request.timestep,
                    initial_scan_ready=False
                )

        return drema_comm_pb2.StreamStatus(
            success=True,
            message="Frame ingested successfully",
            received_timestep=request.timestep,
            initial_scan_ready=False
        )

    def StreamFrames(self, request_iterator, context) -> drema_comm_pb2.StreamStatus:
        last_t = 0
        for obs in request_iterator:
            with self.lock:
                self.received_frames_count += 1
                self.last_timestep = obs.timestep
                last_t = obs.timestep

            if self.on_frame_callback:
                try:
                    self.on_frame_callback(obs)
                except Exception as e:
                    print(f"[DremaServer] Error processing stream frame {obs.timestep}: {e}")

        return drema_comm_pb2.StreamStatus(
            success=True,
            message="Stream ended normally",
            received_timestep=last_t
        )

    def RequestAction(self, request: drema_comm_pb2.RobotState, context) -> drema_comm_pb2.ControlAction:
        if self.on_action_callback:
            try:
                return self.on_action_callback(request)
            except Exception as e:
                print(f"[DremaServer] Error in on_action_callback: {e}")

        # Default fallback: safe zero velocities (hold position)
        n_joints = len(request.joint_positions) if len(request.joint_positions) > 0 else 7
        return drema_comm_pb2.ControlAction(
            timestamp=time.time(),
            timestep=request.timestep,
            joint_velocities=[0.0] * n_joints,
            gripper_action=request.gripper_open,
            safety_stop=True,
            status_message="Default fallback (idle hold)"
        )

    def ResetEpisode(self, request: drema_comm_pb2.ResetRequest, context) -> drema_comm_pb2.ResetResponse:
        with self.lock:
            self.received_frames_count = 0
            self.last_timestep = 0

        success = True
        msg = f"Episode {request.episode_index} for task {request.task_name} reset"
        if self.on_reset_callback:
            try:
                success = self.on_reset_callback(request)
            except Exception as e:
                success = False
                msg = f"Reset callback error: {e}"

        return drema_comm_pb2.ResetResponse(success=success, message=msg)


class DremaGrpcServer:
    """
    gRPC Server wrapper managing thread pool and life cycle.
    """

    def __init__(
        self,
        port: int = 50051,
        max_workers: int = 4,
        on_frame_callback: Optional[Callable[[drema_comm_pb2.FrameObservation], None]] = None,
        on_action_callback: Optional[Callable[[drema_comm_pb2.RobotState], drema_comm_pb2.ControlAction]] = None,
        on_reset_callback: Optional[Callable[[drema_comm_pb2.ResetRequest], bool]] = None
    ):
        self.port = port
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.so_reuseport', 1),
            ]
        )
        self.servicer = DremaInferenceServicer(
            on_frame_callback=on_frame_callback,
            on_action_callback=on_action_callback,
            on_reset_callback=on_reset_callback
        )
        drema_comm_pb2_grpc.add_DremaInferenceServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")

    def start(self):
        self.server.start()
        print(f"✓ DREMA gRPC Inference Server listening on port {self.port}")

    def stop(self, grace: float = 1.0):
        self.server.stop(grace)
        print("✓ DREMA gRPC Server stopped.")
