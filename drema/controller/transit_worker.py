"""
Route planning in a separate process, so the MPPI keeps running while a path is searched. The server sends a
snapshot of the twin (objects, poses, roles, held objects, table) and the planning problem; the worker rebuilds the
scene in its own PyBullet client and plans with the same collision model as the MPPI.

Request: start, goals [n, 7], preferred_goal (optional: the keyframe's chosen posture, tried first at every clearance;
the other goals only when no route to it exists, since the next keyframes start from the posture this one ends in),
clearance_levels (largest first; the route rule of route_clearance lets the path stay closer only while leaving the
start or approaching the goal), budget_s (per goal stage), attempts, seed, shortcut_tries, wrist_weight.
Run as: python -m drema.controller.transit_worker <socket path>
"""
import os
import pickle
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, Optional

import numpy as np

LEVEL_SHARE = (0.5, 0.3, 0.2)


def _send(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack('!Q', len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def _recv(sock: socket.socket) -> Any:
    (n,) = struct.unpack('!Q', _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, n))


class TransitPlannerService:
    """Client side: starts the worker, submits requests, collects results without blocking."""

    def __init__(self, root: str):
        self._path = os.path.join(tempfile.gettempdir(), f"drema_transit_{os.getpid()}.sock")
        if os.path.exists(self._path):
            os.remove(self._path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self._path)
        server.listen(1)
        server.settimeout(60.0)
        self._proc = subprocess.Popen([sys.executable, '-m', 'drema.controller.transit_worker', self._path],
                                      cwd=root, stdout=subprocess.DEVNULL)
        self._sock, _ = server.accept()
        server.close()
        self._results: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        try:
            while True:
                self._results.put(_recv(self._sock))
        except (EOFError, OSError):
            self._results.put({'error': 'transit worker stopped'})

    def submit(self, request: Dict[str, Any]) -> None:
        _send(self._sock, request)

    def poll(self) -> Optional[dict]:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        try:
            self._sock.close()
            self._proc.terminate()
        except Exception:
            pass


def _build(req: Dict[str, Any]):
    from drema.controller.franka_kinematics import FrankaKinematics
    from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
    kin = FrankaKinematics(base_position=np.asarray(req['base'], np.float64), joint_angle_offsets=req['offsets'],
                           ee_offset=req['ee_offset'], joint_limits=req.get('joint_limits'))
    tw = PyBulletDigitalTwin(visualize=False, table_z=float(req['table_z']), robot_urdf_path=req['urdf'],
                             joint_angle_offsets=req['offsets'], ee_offset=req['ee_offset'],
                             joint_limits=req.get('joint_limits'),
                             table_in_collision_costs=req['table_in_costs'],
                             obstacle_convex_decomposition=req['vhacd'][0], vhacd_resolution=req['vhacd'][1],
                             vhacd_concavity=req['vhacd'][2])
    if req['table_bounds'] is not None:
        tw.spawn_scanned_table(table_z=float(req['table_z']), bounds=tuple(req['table_bounds']))
    tw.load_robot(tuple(np.asarray(req['base']).tolist()), joint_positions=list(req['start']))
    for oid, mesh, name, pos, quat, is_target in req['objects']:
        tw.spawn_scanned_mesh_obstacle(mesh, tuple(pos), tuple(quat), name=name, obj_id=oid, is_target=is_target)
    tw.set_object_policies(req['policies'], req['table_policy'])
    for oid, tcp_T_obj in req['held'].items():
        tw.attach_object(oid, np.asarray(tcp_T_obj), kin)
    return tw, kin


def _plan(req: Dict[str, Any]) -> Dict[str, Any]:
    from drema.controller.route_clearance import clearance_matrix
    from drema.controller.transit_planner import plan_transit
    tw, kin = _build(req)
    try:
        ids = [oid for oid, *_ in req['objects'] if oid not in req['held']] + [-1]
        cap = float(max(req['clearance_levels']))

        def dist(Q):
            return clearance_matrix(tw, kin, Q, cap, kappa=req['kappa'], rho=req['rho'], ids=ids)[0]

        start = np.asarray(req['start'], np.float64)
        goals = np.asarray(req['goals'], np.float64)
        preferred = req.get('preferred_goal')
        lo, hi = kin.Q_MIN.astype(np.float64), kin.Q_MAX.astype(np.float64)
        common = dict(shortcut_tries=req['shortcut_tries'], wrist_weight=req['wrist_weight'])
        spent = 0.0
        stages = ([('chosen posture', np.asarray(preferred, np.float64)[None])] if preferred is not None else []) \
            + [('another usable posture', goals)]
        for label, G in stages:
            for level, c in enumerate(req['clearance_levels']):
                budget = req['budget_s'] * LEVEL_SHARE[min(level, len(LEVEL_SHARE) - 1)]
                path, info = plan_transit(dist, start, G, lo, hi, seed=req['seed'] + 37 * level + len(label),
                                          clearance=float(c), attempts=req['attempts'], budget_s=budget, **common)
                spent += info['time_s']
                if path is not None:
                    info.update(time_s=spent, clearance=float(c), goal=label)
                    return {'key': req['key'], 'request_id': req['request_id'], 'path': path, 'info': info}
        return {'key': req['key'], 'request_id': req['request_id'], 'path': None,
                'info': {'time_s': spent, 'levels': list(req['clearance_levels'])}}
    finally:
        tw.close()


def main(path: str) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    while True:
        try:
            req = _recv(sock)
        except (EOFError, OSError):
            return
        try:
            res = _plan(req)
        except Exception as e:
            res = {'key': req.get('key'), 'request_id': req.get('request_id'), 'path': None, 'error': repr(e)}
            print(f"[TRANSIT WORKER] planning failed: {e!r}", file=sys.stderr, flush=True)
        _send(sock, res)


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    main(sys.argv[1])
