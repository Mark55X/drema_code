"""
Convex decomposition of obstacle meshes, shared by the twin backends. Both PyBullet (dynamic bodies) and MuJoCo
(every mesh geom) collide through the convex hull of a mesh, which fills concave shapes such as the free space
under an arch; V-HACD splits the mesh into convex parts that preserve it. Results are cached next to the mesh.
"""

import os
import sys
import tempfile
import time
from typing import List

import pybullet as p


def _cache_path(mesh_path: str, suffix: str) -> str:
    root, _ = os.path.splitext(mesh_path)
    if os.access(os.path.dirname(os.path.abspath(mesh_path)), os.W_OK):
        return root + suffix
    return os.path.join(tempfile.gettempdir(), os.path.basename(root) + suffix)


def _is_fresh(path: str, source: str) -> bool:
    return os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(source)


def convex_decomposition(mesh_path: str, resolution: int = 100000, concavity: float = 0.05) -> str:
    """OBJ with one 'o convex_<k>' object per convex part (PyBullet loads it as a compound shape)."""
    out_path = _cache_path(mesh_path, f".vhacd_r{int(resolution)}_c{float(concavity):g}.obj")
    if _is_fresh(out_path, mesh_path):
        return out_path
    t0 = time.perf_counter()
    # V-HACD prints its progress from C straight to the stdout file descriptor.
    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        p.vhacd(mesh_path, out_path, os.path.join(tempfile.gettempdir(), "drema_vhacd.log"),
                resolution=int(resolution), concavity=float(concavity), maxNumVerticesPerCH=16)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull)
    print(f"[MESH DECOMPOSITION] '{os.path.basename(mesh_path)}' split into convex parts, cached to '{out_path}' "
          f"({time.perf_counter() - t0:.1f}s)")
    return out_path


def convex_parts(mesh_path: str, resolution: int = 100000, concavity: float = 0.05) -> List[str]:
    """One OBJ file per convex part (MuJoCo needs a separate mesh asset per convex geom)."""
    combined = convex_decomposition(mesh_path, resolution, concavity)
    part_dir = os.path.splitext(combined)[0] + "_parts"
    vertices, parts, current = [], [], None
    with open(combined) as f:
        for line in f:
            if line.startswith('o '):
                current = []
                parts.append(current)
            elif line.startswith('v '):
                vertices.append(line.split()[1:4])
            elif line.startswith('f ') and current is not None:
                current.append([int(tok.split('/')[0]) - 1 for tok in line.split()[1:]])
    paths = [os.path.join(part_dir, f"part_{k:03d}.obj") for k in range(len(parts))]
    if all(_is_fresh(path, combined) for path in paths) and paths:
        return paths
    os.makedirs(part_dir, exist_ok=True)
    for path, faces in zip(paths, parts):
        used = sorted({i for face in faces for i in face})
        local = {g: k + 1 for k, g in enumerate(used)}
        with open(path, 'w') as f:
            f.writelines(f"v {' '.join(vertices[g])}\n" for g in used)
            f.writelines("f " + " ".join(str(local[i]) for i in face) + "\n" for face in faces)
    return paths
