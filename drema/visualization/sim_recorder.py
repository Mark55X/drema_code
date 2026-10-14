"""
Client-side video recording of the simulated scene, from a dedicated high-resolution vision sensor placed at
an existing camera pose, so the streaming cameras used by perception are left untouched.
Frames are keyed by the client step counter, which is also the timestep of the server imagination frames; file
names carry a running sequence number, so a client reset (which restarts the counter) never overwrites frames.
"""

import os
import queue
import threading
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .sinks import warn_limited


class SimVideoRecorder:
    """Records while the task is active and for `tail_steps` steps after it stops (e.g. after success)."""

    def __init__(self, out_dir: str, reference_camera: Any, resolution: Tuple[int, int] = (960, 540),
                 every_n: int = 1, tail_steps: int = 40, jpeg_quality: int = 92, view_angle_deg: Optional[float] = None):
        from pyrep.objects.vision_sensor import VisionSensor  # client-only dependency

        self.out_dir = out_dir
        self.every_n = max(1, int(every_n))
        self.tail_steps = max(0, int(tail_steps))
        self.jpeg_quality = int(jpeg_quality)
        self._tail_left = 0
        self._sequence = 0
        os.makedirs(out_dir, exist_ok=True)

        if view_angle_deg is None:
            try:
                view_angle_deg = float(reference_camera.get_perspective_angle())
            except RuntimeError:
                view_angle_deg = 60.0  # some PyRep / CoppeliaSim builds do not expose the parameter
        self.sensor = VisionSensor.create(
            [int(resolution[0]), int(resolution[1])],
            explicit_handling=True,
            view_angle=float(view_angle_deg),
            position=list(reference_camera.get_position()),
            orientation=list(reference_camera.get_orientation())
        )
        self._index = open(os.path.join(out_dir, 'index.csv'), 'w')
        self._index.write('sequence,step,sim_time,file\n')
        self._queue: "queue.Queue" = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        print(f"[CoppeliaClient] Recording simulation video frames to '{out_dir}' at {resolution[0]}x{resolution[1]}")

    def capture(self, step: int, sim_time: float, task_active: bool) -> None:
        if task_active:
            self._tail_left = self.tail_steps
        elif self._tail_left <= 0:
            return
        else:
            self._tail_left -= 1
        if step % self.every_n:
            return
        self.sensor.handle_explicitly()
        rgb = (np.clip(self.sensor.capture_rgb(), 0.0, 1.0) * 255.0).astype(np.uint8)
        self._sequence += 1
        try:
            self._queue.put_nowait((self._sequence, int(step), float(sim_time), rgb))
        except queue.Full:
            pass

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            sequence, step, sim_time, rgb = item
            name = f"sim_{sequence:06d}_s{step:05d}.jpg"
            try:
                cv2.imwrite(os.path.join(self.out_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                self._index.write(f"{sequence},{step},{sim_time:.4f},{name}\n")
                self._index.flush()
            except Exception as e:
                warn_limited(self, f"frame {sequence} not written: {e}")

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._index.close()
