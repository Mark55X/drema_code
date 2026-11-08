"""
ROS2 robot-state bridge, the protocol of Drema2's ros2_joints (`get_joint_state`).

Additions, backward compatible: several JointState topics merged by joint name (arm and gripper are published apart
on the FR3, /franka/joint_states and /fr3_gripper/joint_states), and `get_robot_state`, which adds the hand pose the
robot itself reports (franka_robot_state_broadcaster's current_pose, O_T_EE in the base frame).

    python -m lab_bridges.ros2_joints --port 5003 --topics /franka/joint_states /fr3_gripper/joint_states \
        --pose_topic /franka_robot_state_broadcaster/current_pose
"""

import argparse
import threading
import time
from typing import Any, Dict, List, Optional

from lab_bridges.wire import BridgeClient, error, handle_ping, ok, serve


class JointStateClient(BridgeClient):

    def __init__(self, host: str = "127.0.0.1", port: int = 5003, timeout: float = 2.0):
        super().__init__(host, port, timeout, name="ros2_joints")
        self._robot_state_supported = True

    def get_joint_state(self) -> Dict[str, Any]:
        """{names, positions, velocities, efforts, stamp}."""
        return self.request("get_joint_state")["payload"]

    def get_robot_state(self) -> Dict[str, Any]:
        """get_joint_state plus ee_pose [x, y, z, qx, qy, qz, qw] (None if not published), ee_age and age [s] of the
        stalest joint topic; against a Drema2 bridge (no get_robot_state) the joint state alone, ages None."""
        if self._robot_state_supported:
            resp = self.request("get_robot_state", allow_error=True)
            if resp.get("status") == "ok":
                return resp["payload"]
            if "Unknown type" not in str(resp.get("error_msg", "")):
                raise RuntimeError(f"ros2_joints: {resp.get('error_msg')}")
            self._robot_state_supported = False
        state = self.get_joint_state()
        state.update(ee_pose=None, ee_age=None, age=None)
        return state


class JointStateStore:
    """Joint states of several topics merged by name, ROS-free. `age` is the age of the stalest topic, so an arm
    topic that stops is noticed even while the gripper topic keeps publishing."""

    def __init__(self):
        self._lock = threading.Lock()
        self._joints: Dict[str, Dict[str, float]] = {}
        self._order: List[str] = []
        self._received: Dict[str, float] = {}
        self._stamp = None
        self._ee_pose = None
        self._ee_received = None

    def update(self, topic: str, names, positions, velocities, efforts, stamp: float,
               now: Optional[float] = None) -> None:
        with self._lock:
            for i, name in enumerate(names):
                if name not in self._joints:
                    self._order.append(name)
                self._joints[name] = {
                    "position": float(positions[i]) if i < len(positions) else 0.0,
                    "velocity": float(velocities[i]) if i < len(velocities) else 0.0,
                    "effort": float(efforts[i]) if i < len(efforts) else 0.0,
                }
            self._stamp = stamp
            self._received[topic] = time.monotonic() if now is None else now

    def set_pose(self, pose: List[float], now: Optional[float] = None) -> None:
        with self._lock:
            self._ee_pose = list(pose)
            self._ee_received = time.monotonic() if now is None else now

    def latest(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self._joints:
                return None
            return {
                "names": list(self._order),
                "positions": [self._joints[n]["position"] for n in self._order],
                "velocities": [self._joints[n]["velocity"] for n in self._order],
                "efforts": [self._joints[n]["effort"] for n in self._order],
                "stamp": self._stamp,
                "ee_pose": None if self._ee_pose is None else list(self._ee_pose),
                "ee_age": None if self._ee_received is None else now - self._ee_received,
                "age": now - min(self._received.values()),
            }


def make_node(topics: List[str], pose_topic: Optional[str], store: JointStateStore):
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class RobotStateNode(Node):

        def __init__(self):
            super().__init__("drema_joint_state_bridge")
            for topic in topics:
                self.create_subscription(JointState, topic, lambda msg, t=topic: self._joint_callback(t, msg), 10)
                self.get_logger().info(f"Subscribed to {topic}")
            if pose_topic:
                self.create_subscription(PoseStamped, pose_topic, self._pose_callback, 10)
                self.get_logger().info(f"Subscribed to {pose_topic}")

        def _joint_callback(self, topic, msg):
            store.update(topic, msg.name, msg.position, msg.velocity, msg.effort,
                         msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        def _pose_callback(self, msg):
            p, o = msg.pose.position, msg.pose.orientation
            store.set_pose([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

    return RobotStateNode()


def handle_get_joint_state(state, payload):
    js = state["store"].latest()
    if js is None:
        return error("No joint_state received yet")
    return ok({k: js[k] for k in ("names", "positions", "velocities", "efforts", "stamp")})


def handle_get_robot_state(state, payload):
    js = state["store"].latest()
    if js is None:
        return error("No joint_state received yet")
    return ok(js)


HANDLERS = {"get_joint_state": handle_get_joint_state, "get_robot_state": handle_get_robot_state, "ping": handle_ping}


def main():
    parser = argparse.ArgumentParser(description="ROS2 robot-state bridge for the DREMA lab environment")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--topics", nargs="+", default=["/franka/joint_states", "/fr3_gripper/joint_states"])
    parser.add_argument("--pose_topic", default="/franka_robot_state_broadcaster/current_pose",
                        help="PoseStamped of the hand in the base frame; '' to disable")
    args = parser.parse_args()

    import rclpy
    rclpy.init()
    store = JointStateStore()
    node = make_node(args.topics, args.pose_topic or None, store)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    serve(HANDLERS, {"store": store}, args.host, args.port, "ROS2 JOINTS BRIDGE")


if __name__ == "__main__":
    main()
