#!/usr/bin/env python
"""
Environment-side role providers. Each one can be switched on or off in configs/roles_env.yaml; all of them write
into a RoleManager, whose full snapshot travels to the server with every control request (RobotState), where the
arbitration (only a human relaxes) is enforced.
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import Permission, RoleAssignment, Source
from .vlm import GroundedCommand, SceneObject, VLMBackend


class RoleManager:

    def __init__(self):
        self._lock = threading.Lock()
        self._by_source: Dict[Source, Dict[int, RoleAssignment]] = {}
        self.revision = 0
        self.server_status: Dict[int, Dict[str, Any]] = {}

    def set(self, assignment: RoleAssignment) -> None:
        with self._lock:
            self._by_source.setdefault(assignment.source, {})[assignment.object_id] = assignment
            self.revision += 1

    def clear(self, object_id: int, source: Source) -> bool:
        with self._lock:
            removed = self._by_source.get(source, {}).pop(object_id, None) is not None
            if removed:
                self.revision += 1
            return removed

    def replace_source(self, source: Source, assignments: List[RoleAssignment]) -> None:
        with self._lock:
            self._by_source[source] = {a.object_id: a for a in assignments}
            self.revision += 1

    def reset(self) -> None:
        with self._lock:
            self._by_source.clear()
            self.server_status = {}
            self.revision += 1

    def snapshot(self) -> Tuple[int, List[RoleAssignment]]:
        with self._lock:
            return self.revision, [a for per in self._by_source.values() for a in per.values()]

    def update_server_status(self, object_roles: Any) -> None:
        status = {int(s.object_id): {"role": s.role, "decided_by": s.decided_by,
                                     "hand_contact_allowed": s.hand_contact_allowed} for s in object_roles}
        if status:
            self.server_status = status

    def describe(self, names: Dict[int, str]) -> str:
        _, assignments = self.snapshot()
        lines = ["  Environment assignments:"]
        lines += [f"    {a.object_id} ({names.get(a.object_id, '?')}): {a.permission.name} from {a.source.name}"
                  + (f" - {a.reason}" if a.reason else "") for a in assignments] or ["    (none)"]
        lines.append("  Server roles (effective):")
        lines += [f"    {oid} ({names.get(oid, '?')}): {s['role']} by {s['decided_by']}"
                  for oid, s in sorted(self.server_status.items())] or ["    (no feedback yet)"]
        return "\n".join(lines)


class ConfigRoleProvider:
    """Static rules from the config file: object-name substring -> permission (restrictions only)."""

    def __init__(self, rules: List[Dict[str, Any]], manager: RoleManager, log: Callable[[str], None] = print):
        self.rules = rules or []
        self.manager = manager
        self.log = log

    def apply(self, objects: List[SceneObject]) -> None:
        assignments = []
        for obj in objects:
            for rule in self.rules:
                if str(rule.get("match", "")).lower() in obj.name.lower():
                    perm = Permission.parse(rule.get("permission", "avoid"))
                    if perm == Permission.TOUCH:
                        self.log(f"Config rule '{rule.get('match')}' asks TOUCH for {obj.name}: ignored, only a human relaxes")
                        break
                    assignments.append(RoleAssignment(obj.object_id, perm, Source.CONFIG,
                                                      f"rule '{rule.get('match')}'", float(rule.get("extra_margin", 0.0))))
                    break
        self.manager.replace_source(Source.CONFIG, assignments)
        if assignments:
            self.log("Config roles: " + ", ".join(f"{a.object_id}:{a.permission.name}" for a in assignments))


class VLMSafetyProvider:
    """Asks the VLM, in the background, which objects must not be touched; it can only restrict."""

    def __init__(self, backend: VLMBackend, manager: RoleManager, log: Callable[[str], None] = print):
        self.backend = backend
        self.manager = manager
        self.log = log
        self._busy = threading.Lock()

    def request(self, objects: List[SceneObject], image: Optional[np.ndarray], instruction: str) -> bool:
        if not objects or not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._run, args=(objects, image, instruction), daemon=True).start()
        return True

    def _run(self, objects, image, instruction) -> None:
        try:
            flags = self.backend.safety_flags(objects, image, instruction)
            ids = {o.object_id for o in objects}
            assignments = []
            for f in flags:
                perm = Permission.parse(f.permission) if f.permission else Permission.AVOID
                if f.object_id not in ids or perm == Permission.TOUCH:
                    continue
                assignments.append(RoleAssignment(f.object_id, perm, Source.VLM, f.reason))
            self.manager.replace_source(Source.VLM, assignments)
            self.log("VLM safety check: " + (", ".join(f"{a.object_id}:{a.permission.name} ({a.reason})"
                                                        for a in assignments) or "no restriction"))
        except Exception as e:
            self.log(f"VLM safety check failed: {e}")
        finally:
            self._busy.release()


class HumanRoleProvider:
    """
    Operator messages during the task, grounded to an object by the VLM. Restrictions apply at once; a relaxation
    (TOUCH) applies at once only when the grounding is confident and unambiguous (confirm_relax='uncertain'),
    otherwise it waits for an explicit confirmation; confirm_relax='always' asks every time.
    """

    def __init__(self, backend: Optional[VLMBackend], manager: RoleManager, min_confidence: float = 0.7,
                 confirm_relax: str = "uncertain", log: Callable[[str], None] = print):
        self.backend = backend
        self.manager = manager
        self.min_confidence = float(min_confidence)
        self.confirm_relax = str(confirm_relax).lower()
        self.log = log
        self.pending: Optional[RoleAssignment] = None
        self._busy = threading.Lock()

    def submit_message(self, message: str, objects: List[SceneObject], image: Optional[np.ndarray],
                       instruction: str) -> None:
        if self.backend is None:
            self.log("No VLM configured: use 'role <id> <touch|avoid|fragile|clear>'")
            return
        if not self._busy.acquire(blocking=False):
            self.log("Still interpreting the previous message, try again in a moment")
            return
        threading.Thread(target=self._ground, args=(message, objects, image, instruction), daemon=True).start()

    def _ground(self, message, objects, image, instruction) -> None:
        try:
            cmd = self.backend.ground_command(message, objects, image, instruction)
            self._apply_grounded(cmd, {o.object_id: o.name for o in objects})
        except Exception as e:
            self.log(f"Could not interpret the message: {e}")
        finally:
            self._busy.release()

    def _apply_grounded(self, cmd: GroundedCommand, names: Dict[int, str]) -> None:
        if cmd.ambiguous or cmd.object_id is None:
            options = ", ".join(f"{c} ({names.get(c, '?')})" for c in cmd.candidates) or "none"
            self.log(f"Which object do you mean? Candidates: {options}. "
                     f"Repeat with the id, e.g. 'role <id> {cmd.permission or 'touch'}'")
            return
        if cmd.object_id not in names:
            self.log(f"The VLM answered object {cmd.object_id}, which is not in the scene: ignored")
            return
        if cmd.permission not in ("touch", "avoid", "fragile", "clear"):
            self.log(f"Understood object {cmd.object_id} ({names[cmd.object_id]}) but not what to do with it "
                     f"('{cmd.permission}')")
            return
        needs_confirm = cmd.permission == "touch" and (
            self.confirm_relax == "always" or cmd.confidence < self.min_confidence)
        self.apply(cmd.object_id, cmd.permission, names, reason=cmd.explanation, confirm=needs_confirm,
                   confidence=cmd.confidence)

    def apply(self, object_id: int, permission: str, names: Dict[int, str], reason: str = "",
              confirm: bool = False, confidence: float = 1.0) -> None:
        name = names.get(object_id, "?")
        if permission == "clear":
            removed = self.manager.clear(object_id, Source.HUMAN)
            self.log(f"Human instruction on {object_id} ({name}) {'removed' if removed else 'was not set'}")
            return
        assignment = RoleAssignment(object_id, Permission.parse(permission), Source.HUMAN, reason)
        if assignment.permission == Permission.TOUCH and confirm:
            self.pending = assignment
            self.log(f"You want to allow contact with object {object_id} ({name}) "
                     f"[confidence {confidence:.2f}]. Confirm with 'yes', cancel with 'no'.")
            return
        self.manager.set(assignment)
        verb = "may now touch" if assignment.permission == Permission.TOUCH else f"set to {assignment.permission.name} for"
        self.log(f"Robot {verb} object {object_id} ({name})")

    def confirm(self, accepted: bool) -> None:
        if self.pending is None:
            self.log("Nothing to confirm")
            return
        pending, self.pending = self.pending, None
        if accepted:
            self.manager.set(pending)
            self.log(f"Confirmed: robot may now touch object {pending.object_id}")
        else:
            self.log(f"Cancelled: object {pending.object_id} stays as it was")


__all__ = ["RoleManager", "ConfigRoleProvider", "VLMSafetyProvider", "HumanRoleProvider", "SceneObject"]
