"""
Destinations of the imagination frames. Each sink does its work on its own thread, so publishing from the
MPC thread never waits on disk or network I/O.
"""

import os
import queue
import threading
from typing import Any, Dict, Optional

import numpy as np


class ImaginationSink:
    """Receives the static scene once and one frame per recorded solve."""

    def publish_scene(self, scene: Dict[str, Any]) -> None:
        pass

    def publish(self, frame: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def warn_limited(owner: Any, message: str, limit: int = 3) -> None:
    """Prints a recurring background failure a few times, then only counts it, so a broken sink cannot flood the log."""
    count = getattr(owner, '_warnings', 0) + 1
    owner._warnings = count
    if count <= limit:
        suffix = " (further warnings suppressed)" if count == limit else ""
        print(f"[VISUALIZATION WARNING] {type(owner).__name__}: {message}{suffix}")


class _WorkerSink(ImaginationSink):
    """Runs _handle on a background thread; a full queue drops the newest frame instead of blocking."""

    def __init__(self, queue_size: int, every_n: int = 1):
        self.every_n = max(1, int(every_n))
        self._count = 0
        self.dropped = 0
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def publish_scene(self, scene: Dict[str, Any]) -> None:
        self._queue.put(('scene', scene))

    def publish(self, frame: Dict[str, Any]) -> None:
        self._count += 1
        if (self._count - 1) % self.every_n:
            return
        try:
            self._queue.put_nowait(('frame', frame))
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == 'scene':
                    self._handle_scene(payload)
                else:
                    self._handle_frame(payload)
            except Exception as e:
                warn_limited(self, f"{kind} failed: {e}")

    def _handle_scene(self, scene: Dict[str, Any]) -> None:
        pass

    def _handle_frame(self, frame: Dict[str, Any]) -> None:
        pass


def load_obj_mesh(path: str, cluster_size: Optional[float] = None) -> Optional[tuple]:
    """
    Vertices, triangular faces and per-vertex RGB colors (None when the OBJ has none) of an OBJ. With cluster_size
    [m], vertices are merged on a grid of that size (vertex-clustering decimation), which keeps the shape while
    cutting the face count for lightweight rendering.
    """
    if not path or not os.path.exists(path):
        return None
    vertices, colors, faces = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                values = [float(x) for x in line.split()[1:]]
                vertices.append(values[:3])
                colors.append(values[3:6] if len(values) >= 6 else None)
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    v, fc = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    col = np.asarray(colors, dtype=np.float64) if colors and all(c is not None for c in colors) else None
    if cluster_size:
        cells, inverse = np.unique(np.round(v / cluster_size).astype(np.int64), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse, minlength=len(cells)).astype(np.float64)
        v = np.stack([np.bincount(inverse, weights=v[:, k], minlength=len(cells)) for k in range(3)], axis=1) / counts[:, None]
        if col is not None:
            col = np.stack([np.bincount(inverse, weights=col[:, k], minlength=len(cells))
                            for k in range(3)], axis=1) / counts[:, None]
        fc = inverse[fc]
        fc = fc[(fc[:, 0] != fc[:, 1]) & (fc[:, 1] != fc[:, 2]) & (fc[:, 0] != fc[:, 2])]
        fc = np.unique(np.sort(fc, axis=1), axis=0)
    return v.astype(np.float32), fc, (np.clip(col, 0.0, 1.0).astype(np.float32) if col is not None else None)


class NpzFrameRecorder(_WorkerSink):
    """
    Writes scene.npz once (with the obstacle meshes embedded, so a recording renders on any machine) and one
    frame_<timestep>.npz per recorded solve, for the offline video renderer.
    """

    def __init__(self, out_dir: str, every_n: int = 1, mesh_cluster_size: float = 0.01):
        self.out_dir = out_dir
        self.mesh_cluster_size = float(mesh_cluster_size)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[VISUALIZATION] Imagination frames recorded to '{out_dir}' (every {max(1, int(every_n))} actions)")
        super().__init__(queue_size=512, every_n=every_n)

    def _handle_scene(self, scene: Dict[str, Any]) -> None:
        data = dict(scene)
        for key in [k for k in scene if k.endswith('_mesh_path')]:
            mesh = load_obj_mesh(str(scene[key]), self.mesh_cluster_size)
            if mesh is not None:
                prefix = key[:-len('_mesh_path')]
                data[f"{prefix}_vertices"], data[f"{prefix}_faces"] = mesh[0], mesh[1]
                if mesh[2] is not None:
                    data[f"{prefix}_colors"] = mesh[2]
        np.savez_compressed(os.path.join(self.out_dir, 'scene.npz'), **data)

    def _handle_frame(self, frame: Dict[str, Any]) -> None:
        self._written = getattr(self, '_written', 0) + 1
        name = f"frame_{self._written:05d}_t{int(frame.get('timestep', 0)):05d}.npz"
        np.savez_compressed(os.path.join(self.out_dir, name), **frame)

    def close(self) -> None:
        super().close()
        if self.dropped:
            print(f"[VISUALIZATION WARNING] {self.dropped} imagination frames dropped (writer queue full)")
