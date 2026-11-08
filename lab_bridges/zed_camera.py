"""
ZED camera bridge (ZED SDK), the protocol of Drema2's zed_camera.

Additions, backward compatible: `--serial` opens one of several ZED cameras (one bridge process per camera, each on
its own port), `grab` accepts {"width", "height"} for a frame resized by the SDK (intrinsics rescaled to it), and the
response carries the image timestamp `stamp` [s].

    python -m lab_bridges.zed_camera --port 5002 [--serial 12345678] [--resolution HD720] [--depth_mode NEURAL]
"""

import argparse
from typing import Any, Dict, Optional

import numpy as np

from lab_bridges.wire import BridgeClient, handle_ping, ok, serve


def decode_image(encoded: Dict[str, Any]) -> np.ndarray:
    return np.frombuffer(encoded["data"], dtype=encoded["dtype"]).reshape(encoded["shape"])


def encode_image(image: np.ndarray) -> Dict[str, Any]:
    return {"data": image.tobytes(), "shape": list(image.shape), "dtype": str(image.dtype)}


class ZedCameraClient(BridgeClient):

    def __init__(self, host: str = "127.0.0.1", port: int = 5002, timeout: float = 30.0):
        super().__init__(host, port, timeout, name="zed_camera")

    def grab(self, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
        """{"rgb" HxWx3 uint8, "depth" HxW float32 [m], "intrinsics" 3x3, "extrinsics" {translation, rotation} or
        None, "stamp" [s] or None}; width/height: frame resized by the SDK (both or neither)."""
        payload = {"width": int(width), "height": int(height)} if width and height else {}
        p = self.request("grab", payload)["payload"]
        ext = p.get("extrinsics")
        return {
            "rgb": decode_image(p["rgb"]),
            "depth": decode_image(p["depth"]),
            "intrinsics": np.array(p["intrinsics"], dtype=np.float64),
            "extrinsics": None if ext is None else {"translation": np.array(ext["translation"]),
                                                     "rotation": np.array(ext["rotation"])},
            "stamp": p.get("stamp"),
        }


def make_server_state(serial: Optional[int], resolution: str, depth_mode: str, enable_tracking: bool,
                      depth_scale: float) -> Dict[str, Any]:
    import pyzed.sl as sl

    depth_modes = {name: getattr(sl.DEPTH_MODE, name) for name in
                   ("PERFORMANCE", "QUALITY", "ULTRA", "NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS")
                   if hasattr(sl.DEPTH_MODE, name)}
    resolutions = {name: getattr(sl.RESOLUTION, name) for name in ("HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA")
                   if hasattr(sl.RESOLUTION, name)}
    zed = sl.Camera()
    init = sl.InitParameters(depth_mode=depth_modes.get(depth_mode, sl.DEPTH_MODE.NEURAL),
                             coordinate_units=sl.UNIT.METER, coordinate_system=sl.COORDINATE_SYSTEM.IMAGE)
    init.camera_resolution = resolutions.get(resolution, sl.RESOLUTION.HD720)
    if serial is not None:
        init.set_from_serial_number(int(serial))
    status = zed.open(init)
    assert status == sl.ERROR_CODE.SUCCESS, f"ZED open failed: {repr(status)}"
    info = zed.get_camera_information()
    print(f"[ZED BRIDGE] Camera {info.serial_number} opened: {resolution}, depth_mode={depth_mode}")
    if enable_tracking:
        status = zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        assert status == sl.ERROR_CODE.SUCCESS, f"Tracking failed: {repr(status)}"
    return {"sl": sl, "zed": zed, "runtime": sl.RuntimeParameters(), "image": sl.Mat(), "depth": sl.Mat(),
            "enable_tracking": enable_tracking, "depth_scale": float(depth_scale)}


def handle_grab(state, payload):
    import cv2

    sl, zed = state["sl"], state["zed"]
    status = zed.grab(state["runtime"])
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED grab failed: {repr(status)}")
    size = sl.Resolution(int(payload["width"]), int(payload["height"])) if payload.get("width") else sl.Resolution(0, 0)

    zed.retrieve_image(state["image"], sl.VIEW.LEFT, sl.MEM.CPU, size)
    rgb = cv2.cvtColor(state["image"].get_data(), cv2.COLOR_BGRA2RGB)
    zed.retrieve_measure(state["depth"], sl.MEASURE.DEPTH, sl.MEM.CPU, size)
    depth = np.nan_to_num(state["depth"].get_data().copy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth *= state["depth_scale"]

    # The calibration of the delivered image size: the SDK rescales it with the image
    cam = zed.get_camera_information(sl.Resolution(rgb.shape[1], rgb.shape[0])) \
        .camera_configuration.calibration_parameters.left_cam
    intrinsics = [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]]

    extrinsics = None
    if state["enable_tracking"]:
        pose = sl.Pose()
        zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        extrinsics = {"translation": pose.get_translation().get().tolist(),
                      "rotation": pose.get_rotation_matrix().r.tolist()}
    stamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
    return ok({"rgb": encode_image(rgb), "depth": encode_image(depth), "intrinsics": intrinsics,
               "extrinsics": extrinsics, "stamp": stamp})


HANDLERS = {"grab": handle_grab, "ping": handle_ping}


def main():
    parser = argparse.ArgumentParser(description="ZED camera bridge for the DREMA lab environment")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--serial", type=int, default=None, help="Serial number of the ZED to open (default: first found)")
    parser.add_argument("--resolution", default="HD720", help="HD2K, HD1200, HD1080, HD720, SVGA, VGA")
    parser.add_argument("--depth_mode", default="NEURAL", help="PERFORMANCE, QUALITY, ULTRA, NEURAL_LIGHT, NEURAL, NEURAL_PLUS")
    parser.add_argument("--enable_tracking", action="store_true", help="ZED positional tracking extrinsics (moving camera)")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    args = parser.parse_args()
    state = make_server_state(args.serial, args.resolution, args.depth_mode, args.enable_tracking, args.depth_scale)
    try:
        serve(HANDLERS, state, args.host, args.port, "ZED BRIDGE")
    finally:
        state["zed"].close()


if __name__ == "__main__":
    main()
