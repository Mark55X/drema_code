#!/usr/bin/env python
"""
Keyframe sources for the environment client and the sequencer that drives them.

A keyframe carries what a PerAct action carries: a 6-DoF gripper pose, the gripper state at that pose and the
ignore_collisions flag. Sources:
- task:   Task.get_keyframes() of custom tasks (recomputed every step, so it follows moving targets);
- demo:   keyposes of an RLBench demo, found with PerAct's keypoint discovery (an oracle with the exact action
          distribution PerAct is trained on: it lets every RLBench task be tested without the network);
- policy: an external keyframe policy (e.g. PerAct), queried with the current observation each time the previous
          keyframe is completed; it returns the PerAct action vector [x, y, z, qx, qy, qz, qw, gripper_open,
          ignore_collisions] in the Panda_tip frame, or None when the episode is over.
"""

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class Keyframe:
    pose: List[float]
    name: str = ""
    frame: str = "flange"                       # "tip" (RLBench / PerAct) or "flange" (FrankaKinematics EE)
    gripper_open: Optional[float] = None        # gripper state at this keypose; None = leave it as it is
    ignore_collisions: Optional[bool] = None
    position_tolerance: float = 0.02
    approach_tolerance_deg: float = 10.0
    rotation_tolerance_deg: Optional[float] = None   # full orientation check, tip-frame keyframes only
    approach_funnel: Optional[bool] = None
    allow_contact: bool = False                      # hand contact at the keypose even without ignore_collisions
    precise: bool = False                            # approach pose of a grasp / release: left once the arm has settled

    def contact_flag(self) -> Optional[bool]:
        """Value sent as RobotState.ignore_collisions: the server reads it as 'hand may touch what is at the keypose'."""
        return True if self.allow_contact else self.ignore_collisions

    @classmethod
    def from_dict(cls, d: Dict[str, Any], index: int) -> "Keyframe":
        return cls(
            pose=list(d['pose']),
            name=str(d.get('name', f"kf{index}")),
            frame=str(d.get('frame', 'flange')),
            gripper_open=d.get('gripper_open'),
            ignore_collisions=d.get('ignore_collisions'),
            position_tolerance=float(d.get('position_tolerance', 0.02)),
            approach_tolerance_deg=float(d.get('approach_tolerance_deg', 10.0)),
            rotation_tolerance_deg=d.get('rotation_tolerance_deg'),
            approach_funnel=d.get('approach_funnel')
        )

    @classmethod
    def from_action_vector(cls, action: np.ndarray, name: str, position_tolerance: float = 0.01,
                           rotation_tolerance_deg: float = 10.0) -> "Keyframe":
        """PerAct action layout: [x, y, z, qx, qy, qz, qw, gripper_open, ignore_collisions] (Panda_tip frame)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        quat = a[3:7] / max(np.linalg.norm(a[3:7]), 1e-9)
        return cls(
            pose=[float(x) for x in a[:3]] + [float(x) for x in quat],
            name=name, frame="tip",
            gripper_open=float(a[7] > 0.5),
            ignore_collisions=bool(a[8] > 0.5) if len(a) > 8 else None,
            position_tolerance=position_tolerance,
            approach_tolerance_deg=rotation_tolerance_deg,
            rotation_tolerance_deg=rotation_tolerance_deg
        )


def _quat_z_axis(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


def _quat_angle_deg(q1, q2) -> float:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def keyframe_errors(kf: Keyframe, ee_pose: List[float]) -> Dict[str, float]:
    """Position error [m], approach-axis error [deg] and (tip frame only) full rotation error [deg]."""
    pos_err = float(np.linalg.norm(np.asarray(ee_pose[:3]) - np.asarray(kf.pose[:3])))
    out = {'pos': pos_err, 'axis': 0.0, 'rot': 0.0}
    if len(kf.pose) >= 7 and len(ee_pose) >= 7:
        cos_axis = float(np.dot(_quat_z_axis(ee_pose[3:7]), _quat_z_axis(kf.pose[3:7])))
        out['axis'] = float(np.degrees(np.arccos(np.clip(cos_axis, -1.0, 1.0))))
        # Panda_tip shares its z-axis with the flange, so the axis check is frame independent; the full rotation
        # can only be compared when the keyframe is expressed in the Panda_tip frame like ee_pose. Tip-frame
        # keyframes are yaw symmetric (parallel gripper): the pose turned by 180 deg about the gripper axis counts too.
        if kf.frame == "tip":
            x, y, z, w = kf.pose[3:7]
            flipped = [y, -x, w, -z]  # q * (0, 0, 1, 0): rotation by pi about the local z-axis
            out['rot'] = min(_quat_angle_deg(ee_pose[3:7], kf.pose[3:7]), _quat_angle_deg(ee_pose[3:7], flipped))
    return out


def keyframe_reached(kf: Keyframe, ee_pose: List[float], max_position_tolerance: Optional[float] = None,
                     max_rotation_tolerance_deg: Optional[float] = None) -> bool:
    """The optional maxima tighten the keyframe tolerances (never loosen them)."""
    err = keyframe_errors(kf, ee_pose)
    pos_tol = kf.position_tolerance if max_position_tolerance is None else min(kf.position_tolerance,
                                                                               max_position_tolerance)
    axis_tol = kf.approach_tolerance_deg if max_rotation_tolerance_deg is None else min(kf.approach_tolerance_deg,
                                                                                         max_rotation_tolerance_deg)
    if err['pos'] > pos_tol or err['axis'] > axis_tol:
        return False
    if kf.rotation_tolerance_deg is None or kf.frame != "tip":
        return True
    rot_tol = float(kf.rotation_tolerance_deg)
    if max_rotation_tolerance_deg is not None:
        rot_tol = min(rot_tol, max_rotation_tolerance_deg)
    return err['rot'] <= rot_tol


# -----------------------------------------------------------------------------
# PerAct keypoint discovery (Shridhar et al., peract/helpers/demo_loading_utils.py)
# -----------------------------------------------------------------------------

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1) -> bool:
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
        i < (len(demo) - 2)
        and (obs.gripper_open == demo[i + 1].gripper_open and obs.gripper_open == demo[i - 1].gripper_open
             and demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return stopped_buffer <= 0 and small_delta and (not next_is_not_final) and gripper_state_no_change


def discover_keypoints(demo, stopping_delta: float = 0.1) -> List[int]:
    keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(keypoints) > 1 and (keypoints[-1] - 1) == keypoints[-2]:
        keypoints.pop(-2)
    return keypoints


def keyframes_from_demo(demo, position_tolerance: float = 0.02, rotation_tolerance_deg: float = 15.0) -> List[Keyframe]:
    """
    PerAct targets: gripper pose and state at each keypoint, ignore_collisions of the step before it.
    The approach cone is kept only on grasp keyframes (gripper closing there): elsewhere it would bend free motion.
    """
    frames = []
    prev_open = float(demo[0].gripper_open)
    for n, i in enumerate(discover_keypoints(demo)):
        obs, prev = demo[i], demo[max(0, i - 1)]
        action = np.concatenate([obs.gripper_pose, [float(obs.gripper_open)], [float(prev.ignore_collisions)]])
        kf = Keyframe.from_action_vector(action, f"demo_kp{n}@{i}", position_tolerance, rotation_tolerance_deg)
        kf.approach_funnel = prev_open > 0.5 and kf.gripper_open < 0.5
        prev_open = kf.gripper_open
        frames.append(kf)
    # The pre-grasp keypose sits a couple of centimetres above the object to grasp, inside its collision margin.
    # Grasps and releases start from the keypose before them: the short final motion cannot fix a lateral offset
    # left there once the fingers or the held object touch.
    for kf, nxt in zip(frames[:-1], frames[1:]):
        if nxt.approach_funnel:
            kf.allow_contact = True
        kf.precise = (nxt.gripper_open > 0.5) != (kf.gripper_open > 0.5)
    return frames


# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

class KeyframeSource:
    """keyframe(index, observe) -> the index-th keyframe, or None once the source has no more keyframes."""

    def keyframe(self, index: int, observe: Callable[[], Any]) -> Optional[Keyframe]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class TaskKeyframeSource(KeyframeSource):

    def __init__(self, task_inst: Any):
        self.task_inst = task_inst

    def keyframe(self, index, observe):
        frames = self.task_inst.get_keyframes()
        return Keyframe.from_dict(frames[index], index) if index < len(frames) else None


class StaticKeyframeSource(KeyframeSource):

    def __init__(self, frames: List[Keyframe]):
        self.frames = list(frames)

    def keyframe(self, index, observe):
        return self.frames[index] if index < len(self.frames) else None


class PolicyKeyframeSource(KeyframeSource):
    """
    External keyframe policy. `act` receives the RLBench Observation and returns the PerAct action vector, or None
    when it considers the episode finished. Load one with --policy module:function; the function is called once
    with the client (to set itself up) and must return the `act` callable.
    """

    def __init__(self, act: Callable[[Any], Optional[np.ndarray]], max_keyframes: int = 25):
        self.act = act
        self.max_keyframes = int(max_keyframes)
        self.cache: Dict[int, Optional[Keyframe]] = {}

    def keyframe(self, index, observe):
        if index >= self.max_keyframes:
            return None
        if index not in self.cache:
            obs = observe()
            action = self.act(obs)
            kf = None if action is None else Keyframe.from_action_vector(action, f"policy_kf{index}")
            if kf is not None:
                # Approach cone on grasps only, as for demo keyframes: elsewhere it would bend free motion
                kf.approach_funnel = float(getattr(obs, 'gripper_open', 1.0)) > 0.5 and kf.gripper_open < 0.5
            self.cache[index] = kf
        return self.cache[index]

    def reset(self):
        self.cache.clear()

    @staticmethod
    def load(spec: str, client: Any) -> "PolicyKeyframeSource":
        module_name, func_name = spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), func_name)
        return PolicyKeyframeSource(factory(client))


class KeyframeSequencer:
    """
    Next-best-pose execution as RLBench runs PerAct actions: move until the keyframe tolerances are met, then set
    the gripper to the keyframe state and wait for it, then move on. The last keyframe is held.
    Keyframes where the gripper changes state get tighter tolerances: a grasp or a release closed a couple of
    centimetres off the keypose misses the object, while free-space keyframes only need to be passed through.
    With settling, those keyframes (and the precise approach poses before them) are reached only once the arm stops
    closing in, not on the first step inside the tolerance: fingers closed on the way in grasp the object off centre.
    """

    def __init__(self, source: KeyframeSource, log: Callable[[str], None] = print,
                 gripper_change_position_tolerance: float = 0.01, gripper_change_rotation_tolerance_deg: float = 10.0,
                 settle_on_gripper_change: bool = False, settle_on_precise_approach: bool = False,
                 settle_position_tolerance: float = 0.003, settle_progress: float = 0.001, settle_steps: int = 10,
                 settle_max_steps: int = 60):
        """
        :param settle_on_gripper_change: Actuate the gripper only once the arm has settled on the keypose.
        :param settle_on_precise_approach: Precise keyframes (approach poses of a grasp / release) are left only once
            the arm has settled on them, within their own tolerance (an approach pose the arm cannot get closer to
            is still left).
        :param settle_position_tolerance: Settled at once within this distance [m] of the keypose.
        :param settle_progress / settle_steps: Settled when the distance shrank less than settle_progress [m] over
            the last settle_steps steps.
        :param settle_max_steps: Steps inside the tolerance after which the keyframe counts as reached anyway.
        """
        self.source = source
        self.log = log
        self.gripper_change_position_tolerance = float(gripper_change_position_tolerance)
        self.gripper_change_rotation_tolerance_deg = float(gripper_change_rotation_tolerance_deg)
        self.settle_on_gripper_change = bool(settle_on_gripper_change)
        self.settle_on_precise_approach = bool(settle_on_precise_approach)
        self.settle_position_tolerance = float(settle_position_tolerance)
        self.settle_progress = float(settle_progress)
        self.settle_steps = max(1, int(settle_steps))
        self.settle_max_steps = max(self.settle_steps, int(settle_max_steps))
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.phase = "move"
        self.finished = False
        self._settle_errors: List[float] = []
        self.source.reset()
        self._current: Optional[Keyframe] = None

    def current(self, observe: Callable[[], Any]) -> Optional[Keyframe]:
        kf = self.source.keyframe(self.index, observe)
        if kf is None and self.index > 0:
            self.finished = True
            self.index -= 1
            kf = self.source.keyframe(self.index, observe)
        self._current = kf
        return kf

    def step(self, ee_pose: List[float], gripper: Any) -> None:
        """Advances the sequence; `gripper` exposes command(open_amount), done and is_open()."""
        kf = self._current
        if kf is None or self.finished:
            return
        gripper_change = kf.gripper_open is not None and (kf.gripper_open > 0.5) != gripper.is_open()
        precise = self.settle_on_precise_approach and kf.precise
        tight = (self.gripper_change_position_tolerance, self.gripper_change_rotation_tolerance_deg) \
            if gripper_change else (None, None)
        settle = precise or (gripper_change and self.settle_on_gripper_change)
        if self.phase == "move" and keyframe_reached(kf, ee_pose, *tight):
            err = keyframe_errors(kf, ee_pose)
            if settle and not self._settled(err['pos']):
                return
            settled = f", settled after {len(self._settle_errors)} steps" if settle else ""
            self.log(f"Keyframe '{kf.name}' reached (pos {err['pos'] * 100:.1f} cm, axis {err['axis']:.1f} deg"
                     + (f", rot {err['rot']:.1f} deg" if kf.frame == "tip" else "") + f"{settled})")
            if kf.gripper_open is not None and (kf.gripper_open > 0.5) != gripper.is_open():
                gripper.command(kf.gripper_open)
                self.phase = "gripper"
                self.log(f"Gripper -> {'open' if kf.gripper_open > 0.5 else 'close'}")
            else:
                self._advance()
        elif self.phase == "gripper" and gripper.done:
            self._advance()
        elif self.phase == "move":
            self._settle_errors = []

    def _settled(self, pos_err: float) -> bool:
        self._settle_errors.append(pos_err)
        n = len(self._settle_errors)
        if pos_err <= self.settle_position_tolerance or n >= self.settle_max_steps:
            return True
        return n > self.settle_steps and self._settle_errors[-self.settle_steps - 1] - pos_err < self.settle_progress

    def _advance(self) -> None:
        self.index += 1
        self.phase = "move"
        self._settle_errors = []
