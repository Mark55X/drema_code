"""
ROS2 camera bridge, the protocol of Drema2's ros2_camera: the latest RGB, depth and camera_info of one camera,
zlib-compressed on `grab`. Any RGB-D camera with a ROS2 driver (ZED wrapper, RealSense, ...) works through it; depth
must be registered to the RGB image. Addition, backward compatible: the response carries the RGB header stamp [s].

    python -m lab_bridges.ros2_camera --port 5004 --rgb_topic /zed/zed_node/left/image_rect_color \
        --depth_topic /zed/zed_node/depth/depth_registered --camera_info_topic /zed/zed_node/left/camera_info
"""

import argparse
import threading
import zlib
from typing import Any, Dict, Optional

import numpy as np

from lab_bridges.wire import BridgeClient, error, handle_ping, ok, serve

# ROS2 Image encoding -> (numpy dtype, channels)
ENCODING_INFO = {
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def _decompress(info: Dict[str, Any]) -> np.ndarray:
    return np.frombuffer(zlib.decompress(info["data"]), dtype=info["dtype"]).reshape(info["shape"])


class Ros2CameraClient(BridgeClient):

    def __init__(self, host: str = "127.0.0.1", port: int = 5004, timeout: float = 30.0):
        super().__init__(host, port, timeout, name="ros2_camera")

    def grab(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """{"rgb", "depth" [m], "intrinsics", "extrinsics": None, "stamp"}, or None while no complete frame arrived;
        the frame comes at the driver's resolution (width / height are ignored, the caller resizes)."""
        resp = self.request("grab", allow_error=True)
        if resp.get("status") == "error":
            return None
        p = resp["payload"]
        return {"rgb": _decompress(p["rgb"]), "depth": _decompress(p["depth"]),
                "intrinsics": np.array(p["intrinsics"], dtype=np.float64), "extrinsics": None,
                "stamp": p.get("stamp")}


def _image_msg_to_numpy(msg) -> np.ndarray:
    info = ENCODING_INFO.get(msg.encoding)
    assert info is not None, f"Unsupported image encoding: {msg.encoding}"
    dtype, channels = info
    if channels == 1:
        return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)


def _to_rgb(arr: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "bgr8":
        return arr[:, :, ::-1].copy()
    if encoding == "bgra8":
        return arr[:, :, 2::-1].copy()
    if encoding == "rgba8":
        return arr[:, :, :3].copy()
    return arr.copy()


def make_node(rgb_topic: str, depth_topic: str, camera_info_topic: str, depth_scale: float):
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    class CameraNode(Node):

        def __init__(self):
            super().__init__("drema_camera_bridge")
            self._lock = threading.Lock()
            self._rgb = self._depth = self._intrinsics = None
            self._stamp = None
            self.create_subscription(Image, rgb_topic, self._rgb_callback, 10)
            self.create_subscription(Image, depth_topic, self._depth_callback, 10)
            self.create_subscription(CameraInfo, camera_info_topic, self._info_callback, 10)
            self.get_logger().info(f"RGB {rgb_topic} | depth {depth_topic} | info {camera_info_topic}")

        def _rgb_callback(self, msg):
            rgb = _to_rgb(_image_msg_to_numpy(msg), msg.encoding)
            with self._lock:
                self._rgb = rgb
                self._stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        def _depth_callback(self, msg):
            depth = _image_msg_to_numpy(msg).astype(np.float32)
            if msg.encoding in ("16UC1", "mono16"):
                depth /= depth_scale
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            with self._lock:
                self._depth = depth

        def _info_callback(self, msg):
            with self._lock:
                self._intrinsics = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        def get_latest(self):
            with self._lock:
                if self._rgb is None or self._depth is None or self._intrinsics is None:
                    return None
                return {"rgb": self._rgb.copy(), "depth": self._depth.copy(),
                        "intrinsics": self._intrinsics.copy(), "stamp": self._stamp}

    return CameraNode()


def handle_grab(state, payload):
    frame = state["node"].get_latest()
    if frame is None:
        return error("No complete frame received yet (waiting for RGB, depth, and camera_info)")

    def pack(a):
        return {"data": zlib.compress(a.tobytes()), "shape": list(a.shape), "dtype": str(a.dtype)}

    return ok({"rgb": pack(frame["rgb"]), "depth": pack(frame["depth"]),
               "intrinsics": frame["intrinsics"].tolist(), "stamp": frame["stamp"]})


HANDLERS = {"grab": handle_grab, "ping": handle_ping}


def main():
    parser = argparse.ArgumentParser(description="ROS2 camera bridge for the DREMA lab environment")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--rgb_topic", default="/zed/zed_node/left/image_rect_color")
    parser.add_argument("--depth_topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--camera_info_topic", default="/zed/zed_node/left/camera_info")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="16-bit depth units per meter")
    args = parser.parse_args()

    import rclpy
    rclpy.init()
    node = make_node(args.rgb_topic, args.depth_topic, args.camera_info_topic, args.depth_scale)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    serve(HANDLERS, {"node": node}, args.host, args.port, "ROS2 CAMERA BRIDGE")


if __name__ == "__main__":
    main()
