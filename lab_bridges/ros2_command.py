"""
ROS2 command bridge: the arm and gripper commands of the DREMA lab environment onto the robot's ROS2 controllers.
Drema2 has no counterpart (there robot-stack drives the robot and Drema2 only mirrors /joint_states).

Arm. The client sends, at its control rate, a joint position reference (--interface position) or joint velocities
(--interface velocity). A timer at --publish_hz turns them into a stream for the controller topic: the published
joint velocity follows the commanded one (for a reference: the speed that reaches it in one command period) under
per-joint speed and acceleration bounds, and the published reference integrates it. Commands stop for --watchdog_s,
or `hold` -> the velocity ramps to zero and the reference stays where it stopped. Nothing is published before the
first command, so the bridge does not fight another controller at start-up; `release` stops publishing again once
the arm is still. The controller on the robot PC must accept the topic at --publish_hz and interpolate it to its
1 kHz loop (libfranka aborts on discontinuous references): a ros2_control forward command controller alone does not.

Gripper. franka_gripper's Move (open) and Grasp (close) actions; the width comes from the gripper joint states.

    python -m lab_bridges.ros2_command --port 5005 --interface position \
        --topic /joint_position_controller/commands --msg_type float64_multi_array
"""

import argparse
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from lab_bridges.wire import BridgeClient, handle_ping, ok, serve

FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_Q_MIN = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
FR3_Q_MAX = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
FR3_FINGERS = ["fr3_finger_joint1", "fr3_finger_joint2"]


class CommandClient(BridgeClient):

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, timeout: float = 1.0):
        super().__init__(host, port, timeout, name="ros2_command")

    def command(self, values: Sequence[float]) -> Dict[str, Any]:
        """Joint reference [rad] or velocities [rad/s], as the bridge's --interface."""
        return self.request("command", {"values": [float(v) for v in values]})["payload"]

    def hold(self) -> None:
        self.request("hold")

    def release(self) -> None:
        self.request("release")

    def gripper(self, open_gripper: bool) -> Dict[str, Any]:
        return self.request("gripper", {"open": bool(open_gripper)})["payload"]

    def status(self) -> Dict[str, Any]:
        """{interface, active, holding, stale, q_measured, published, gripper: {width, moving, last_result}}."""
        return self.request("status")["payload"]


class ArmStream:
    """The arm part, ROS-free so it can be tested: `command` / `hold` from the socket, `tick` from the timer."""

    def __init__(self, interface: str, publish_hz: float, watchdog_s: float, max_velocity: Sequence[float],
                 max_acceleration: Sequence[float], q_min: Sequence[float], q_max: Sequence[float],
                 limit_margin: float = 0.05, max_reference_jump: float = 0.25):
        assert interface in ("position", "velocity"), interface
        self.interface = interface
        self.dt = 1.0 / float(publish_hz)
        self.watchdog_s = float(watchdog_s)
        self.v_max = np.asarray(max_velocity, dtype=np.float64)
        self.a_max = np.asarray(max_acceleration, dtype=np.float64)
        self.q_min = np.asarray(q_min, dtype=np.float64) + limit_margin
        self.q_max = np.asarray(q_max, dtype=np.float64) - limit_margin
        self.max_reference_jump = float(max_reference_jump)
        self.lock = threading.Lock()
        self.active = False
        self.holding = False
        self.q_measured: Optional[np.ndarray] = None
        self.q_pub: Optional[np.ndarray] = None
        self.v_pub = np.zeros(len(self.v_max))
        self.target: Optional[np.ndarray] = None
        self.last_command = 0.0
        self.command_period = 0.05

    def set_measured(self, q: Sequence[float]) -> None:
        with self.lock:
            self.q_measured = np.asarray(q, dtype=np.float64)

    def command(self, values: Sequence[float], now: Optional[float] = None) -> Dict[str, Any]:
        now = time.monotonic() if now is None else now
        values = np.asarray(values, dtype=np.float64)
        with self.lock:
            if values.shape != self.v_max.shape:
                raise ValueError(f"expected {self.v_max.size} values, got {values.size}")
            if not self.active:
                if self.q_measured is None:
                    raise RuntimeError("no measured joint state yet: refusing to start the stream")
                self.q_pub = self.q_measured.copy()
                self.v_pub[:] = 0.0
                self.active = True
            if self.interface == "position":
                jump = float(np.max(np.abs(values - self.q_pub)))
                if jump > self.max_reference_jump:
                    raise ValueError(f"reference {jump:.3f} rad away from the published one (max "
                                     f"{self.max_reference_jump:.3f}): rejected")
            if self.last_command > 0.0:
                period = now - self.last_command
                self.command_period = float(np.clip(0.8 * self.command_period + 0.2 * period, self.dt, self.watchdog_s))
            self.target = values
            self.last_command = now
            self.holding = False
            return {"published": self.q_pub.tolist()}

    def hold(self) -> None:
        with self.lock:
            self.holding = True

    def release(self) -> bool:
        """Stops publishing once the arm is still; False while it is still moving (hold first)."""
        with self.lock:
            if np.max(np.abs(self.v_pub)) > 1e-3:
                self.holding = True
                return False
            self.active = False
            self.target = None
            return True

    def stale(self, now: float) -> bool:
        return now - self.last_command > self.watchdog_s

    def tick(self, now: Optional[float] = None) -> Optional[np.ndarray]:
        """Advances the stream by one publish period; the values to publish, or None while inactive."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.active:
                return None
            if self.holding or self.target is None or self.stale(now):
                v_des = np.zeros_like(self.v_pub)
            elif self.interface == "position":
                v_des = (self.target - self.q_pub) / self.command_period
            else:
                v_des = self.target.copy()
            v_des = np.clip(v_des, -self.v_max, self.v_max)
            self.v_pub += np.clip(v_des - self.v_pub, -self.a_max * self.dt, self.a_max * self.dt)
            if self.interface == "position":
                step = self.v_pub * self.dt
                if self.target is not None and not self.holding:
                    remaining = self.target - self.q_pub
                    overshoot = np.abs(step) > np.abs(remaining)
                    step[overshoot] = remaining[overshoot]
                # A reference already past the margin may stay there, never jump back: it only cannot go further
                q_next = np.clip(self.q_pub + step, np.minimum(self.q_min, self.q_pub), np.maximum(self.q_max, self.q_pub))
                self.v_pub = (q_next - self.q_pub) / self.dt
                self.q_pub = q_next
                return self.q_pub.copy()
            # velocity interface: never drive a joint further past its limit margin
            if self.q_measured is not None:
                outward = ((self.q_measured <= self.q_min) & (self.v_pub < 0)) | \
                          ((self.q_measured >= self.q_max) & (self.v_pub > 0))
                self.v_pub[outward] = 0.0
            return self.v_pub.copy()


def make_node(args, arm: ArmStream):
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    try:
        from franka_msgs.action import Grasp, Move
    except ImportError:
        Grasp = Move = None

    class CommandNode(Node):

        def __init__(self):
            super().__init__("drema_command_bridge")
            msg_cls = JointState if args.msg_type == "joint_state" else Float64MultiArray
            self.pub = self.create_publisher(msg_cls, args.topic, 10)
            for topic in args.joint_state_topics:
                self.create_subscription(JointState, topic, self._joint_callback, 10)
            self.create_timer(arm.dt, self._tick)
            self.gripper_lock = threading.Lock()
            self.finger_width = None
            self.gripper_moving = False
            self.gripper_result = None
            self.gripper_goal_time = 0.0
            if Move is None:
                self.get_logger().warn("franka_msgs not found: gripper commands disabled")
                self.move_client = self.grasp_client = None
            else:
                self.move_client = ActionClient(self, Move, args.gripper_move_action)
                self.grasp_client = ActionClient(self, Grasp, args.gripper_grasp_action)
            self.get_logger().info(f"Arm {args.interface} stream -> {args.topic} ({args.msg_type}) at {args.publish_hz} Hz")

        def _joint_callback(self, msg):
            pos = dict(zip(msg.name, msg.position))
            if all(n in pos for n in args.joint_names):
                arm.set_measured([pos[n] for n in args.joint_names])
            if all(n in pos for n in args.finger_joint_names):
                with self.gripper_lock:
                    self.finger_width = float(sum(pos[n] for n in args.finger_joint_names))

        def _tick(self):
            values = arm.tick()
            if values is None:
                return
            if args.msg_type == "joint_state":
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = list(args.joint_names)
                if arm.interface == "position":
                    msg.position = values.tolist()
                else:
                    msg.velocity = values.tolist()
            else:
                msg = Float64MultiArray()
                msg.data = values.tolist()
            self.pub.publish(msg)

        def gripper(self, open_gripper: bool) -> Dict[str, Any]:
            if self.move_client is None:
                raise RuntimeError("gripper disabled (franka_msgs not available)")
            if open_gripper:
                client, goal = self.move_client, Move.Goal()
                goal.width, goal.speed = float(args.gripper_open_width), float(args.gripper_speed)
            else:
                client, goal = self.grasp_client, Grasp.Goal()
                goal.width, goal.speed, goal.force = 0.0, float(args.gripper_speed), float(args.gripper_force)
                goal.epsilon.inner = goal.epsilon.outer = float(args.gripper_open_width)
            if not client.wait_for_server(timeout_sec=1.0):
                raise RuntimeError(f"gripper action server not available ({'move' if open_gripper else 'grasp'})")
            with self.gripper_lock:
                self.gripper_moving = True
                self.gripper_result = None
                self.gripper_goal_time = time.monotonic()
            client.send_goal_async(goal).add_done_callback(self._goal_accepted)
            return {"accepted": True}

        def _goal_accepted(self, future):
            handle = future.result()
            if handle is None or not handle.accepted:
                with self.gripper_lock:
                    self.gripper_moving, self.gripper_result = False, "rejected"
                return
            handle.get_result_async().add_done_callback(self._goal_done)

        def _goal_done(self, future):
            result = getattr(future.result(), "result", None)
            with self.gripper_lock:
                self.gripper_moving = False
                self.gripper_result = "success" if getattr(result, "success", False) else \
                    f"failed: {getattr(result, 'error', '')}"

        def gripper_state(self) -> Dict[str, Any]:
            with self.gripper_lock:
                return {"width": self.finger_width, "moving": self.gripper_moving, "last_result": self.gripper_result}

    return CommandNode()


def handle_command(state, payload):
    return ok(state["arm"].command(payload["values"]))


def handle_hold(state, payload):
    state["arm"].hold()
    return ok()


def handle_release(state, payload):
    return ok({"released": state["arm"].release()})


def handle_gripper(state, payload):
    return ok(state["node"].gripper(bool(payload["open"])))


def handle_status(state, payload):
    arm: ArmStream = state["arm"]
    with arm.lock:
        status = {"interface": arm.interface, "active": arm.active, "holding": arm.holding,
                  "stale": arm.active and arm.stale(time.monotonic()),
                  "q_measured": None if arm.q_measured is None else arm.q_measured.tolist(),
                  "published": None if arm.q_pub is None else arm.q_pub.tolist()}
    status["gripper"] = state["node"].gripper_state()
    return ok(status)


HANDLERS = {"command": handle_command, "hold": handle_hold, "release": handle_release, "gripper": handle_gripper,
            "status": handle_status, "ping": handle_ping}


def main():
    parser = argparse.ArgumentParser(description="ROS2 command bridge (arm + gripper) for the DREMA lab environment")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--interface", choices=["position", "velocity"], default="position")
    parser.add_argument("--topic", required=True, help="Command topic of the arm controller")
    parser.add_argument("--msg_type", choices=["float64_multi_array", "joint_state"], default="float64_multi_array")
    parser.add_argument("--publish_hz", type=float, default=250.0)
    parser.add_argument("--watchdog_s", type=float, default=0.2, help="No command for this long -> hold")
    parser.add_argument("--max_velocity", type=float, nargs=7, default=[0.8] * 7, help="[rad/s] per joint")
    parser.add_argument("--max_acceleration", type=float, nargs=7, default=[3.0] * 7, help="[rad/s^2] per joint")
    parser.add_argument("--limit_margin", type=float, default=0.05, help="[rad] kept from the FR3 joint limits")
    parser.add_argument("--max_reference_jump", type=float, default=0.25,
                        help="[rad] a position command farther than this from the published reference is rejected")
    parser.add_argument("--joint_names", nargs=7, default=FR3_JOINTS)
    parser.add_argument("--finger_joint_names", nargs=2, default=FR3_FINGERS)
    parser.add_argument("--joint_state_topics", nargs="+", default=["/franka/joint_states", "/fr3_gripper/joint_states"])
    parser.add_argument("--gripper_move_action", default="/fr3_gripper/move")
    parser.add_argument("--gripper_grasp_action", default="/fr3_gripper/grasp")
    parser.add_argument("--gripper_open_width", type=float, default=0.08)
    parser.add_argument("--gripper_speed", type=float, default=0.1)
    parser.add_argument("--gripper_force", type=float, default=20.0)
    args = parser.parse_args()

    arm = ArmStream(args.interface, args.publish_hz, args.watchdog_s, args.max_velocity, args.max_acceleration,
                    FR3_Q_MIN, FR3_Q_MAX, args.limit_margin, args.max_reference_jump)
    import rclpy
    rclpy.init()
    node = make_node(args, arm)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    serve(HANDLERS, {"arm": arm, "node": node}, args.host, args.port, "ROS2 COMMAND BRIDGE")


if __name__ == "__main__":
    main()
