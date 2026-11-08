"""
Request/response over TCP with the framing of Drema2's camera and joint bridges, so either side can talk to the other:
4-byte big-endian length + msgpack({"type", "payload"}) -> msgpack({"status": "ok", "payload"} | {"status": "error",
"error_msg"}).
"""

import socket
import struct
import threading
from typing import Any, Callable, Dict, Optional

import msgpack

Handler = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def _recvall(conn: socket.socket, length: int) -> Optional[bytearray]:
    buf = bytearray(length)
    view = memoryview(buf)
    received = 0
    while received < length:
        n = conn.recv_into(view[received:])
        if n == 0:
            return None
        received += n
    return buf


def recv_msg(conn: socket.socket) -> Optional[Dict[str, Any]]:
    raw_len = _recvall(conn, 4)
    if not raw_len:
        return None
    data = _recvall(conn, struct.unpack(">I", raw_len)[0])
    if data is None:
        return None
    return msgpack.unpackb(data, raw=False)


def send_msg(conn: socket.socket, message: Dict[str, Any]) -> None:
    packed = msgpack.packb(message, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(packed)) + packed)


def ok(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"status": "ok", "payload": payload or {}}


def error(message: str) -> Dict[str, Any]:
    return {"status": "error", "error_msg": message}


def handle_ping(state, payload):
    return ok({"message": "pong"})


def serve(handlers: Dict[str, Handler], state: Dict[str, Any], host: str, port: int, name: str) -> None:
    """Accepts clients forever, one thread each; a handler exception is answered as an error, not a crash."""

    def client_loop(conn, addr):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[{name}] Client connected: {addr}")
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                handler = handlers.get(req.get("type"))
                if handler is None:
                    resp = error(f"Unknown type {req.get('type')}")
                else:
                    try:
                        resp = handler(state, req.get("payload", {}) or {})
                    except Exception as e:
                        resp = error(f"{type(e).__name__}: {e}")
                send_msg(conn, resp)
        except OSError:
            pass
        finally:
            print(f"[{name}] Client disconnected: {addr}")
            conn.close()

    print(f"[{name}] Listening on {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)
        while True:
            conn, addr = s.accept()
            threading.Thread(target=client_loop, args=(conn, addr), daemon=True).start()


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    """Connects lazily and reconnects on the next request after a socket error."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, timeout: float = 10.0, name: str = "bridge"):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.name = name
        self.conn: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.close()
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(self.timeout)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.connect((self.host, self.port))
        self.conn = conn

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None

    def request(self, msg_type: str, payload: Optional[Dict[str, Any]] = None,
                allow_error: bool = False) -> Dict[str, Any]:
        """The whole response; raises BridgeError on an error status unless allow_error, OSError on the socket."""
        with self._lock:
            try:
                if self.conn is None:
                    self.connect()
                send_msg(self.conn, {"type": msg_type, "payload": payload or {}})
                resp = recv_msg(self.conn)
            except OSError:
                self.close()
                raise
            if resp is None:
                self.close()
                raise ConnectionError(f"{self.name} at {self.host}:{self.port} closed the connection")
        if resp.get("status") == "error" and not allow_error:
            raise BridgeError(f"{self.name}: {resp.get('error_msg', 'unknown error')}")
        return resp

    def ping(self) -> bool:
        try:
            return self.request("ping").get("status") == "ok"
        except Exception:
            return False
