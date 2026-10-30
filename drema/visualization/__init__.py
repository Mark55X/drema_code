"""
Visualization and recording of the controller, kept out of the control path: the server only calls
VisualizationHub.before_solve / after_solve, and every component is disabled unless configured.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional

from .imagination import ImaginationRecorder
from .mpc_dump import MpcInputDumper
from .sinks import ImaginationSink, NpzFrameRecorder


class VisualizationHub:
    """Fan-out of the per-action hooks to the enabled components."""

    def __init__(self, controller: Any, digital_twin: Any, dumper: Optional[MpcInputDumper] = None,
                 imagination: Optional[ImaginationRecorder] = None):
        self.controller = controller
        self.twin = digital_twin
        self.dumper = dumper
        self.imagination = imagination
        self._pending_dump: Optional[Dict[str, Any]] = None
        if imagination is not None:
            imagination.attach()

    @property
    def active(self) -> bool:
        return self.dumper is not None or self.imagination is not None

    def before_solve(self, robot_state: Any) -> None:
        if self.dumper is not None:
            self._pending_dump = self.dumper.snapshot(robot_state, self.controller, self.twin)
        if self.imagination is not None:
            self.imagination.begin_action(robot_state)

    def after_solve(self, robot_state: Any, action: Any) -> None:
        if self.dumper is not None and self._pending_dump is not None:
            self.dumper.write(self._pending_dump, action, self.controller)
            self._pending_dump = None

    def close(self) -> None:
        if self.imagination is not None:
            self.imagination.close()


def build_visualization_hub(
    config: Any,
    controller: Any,
    digital_twin: Any,
    viser_server: Optional[Any] = None,
    scene_info: Optional[Callable[[], Dict[str, Any]]] = None,
    roles_status: Optional[Callable[[], List[Dict[str, Any]]]] = None
) -> VisualizationHub:
    """
    Reads controller.debug_dump_* and visualization.imagination.* from the config:
    live_overlay (viser), record_dir (frames for the offline video), max_rollouts, ghost_count.
    """
    ctrl_cfg = config.get_nested("controller", {}) or {}
    horizon, dt = int(ctrl_cfg.get("horizon", 15)), float(ctrl_cfg.get("dt", 0.05))

    dumper = None
    dump_root = str(ctrl_cfg.get("debug_dump_dir", "") or "")
    if dump_root:
        dumper = MpcInputDumper(dump_root, interval=int(ctrl_cfg.get("debug_dump_interval", 1)), horizon=horizon, dt=dt)

    img_cfg = config.get_nested("visualization.imagination", {}) or {}
    sinks = []
    if bool(img_cfg.get("live_overlay", False)):
        if viser_server is None:
            print("[VISUALIZATION WARNING] live_overlay requested but viser is disabled: overlay skipped")
        else:
            from .viser_overlay import ViserImaginationOverlay  # viser is an optional dependency
            sinks.append(ViserImaginationOverlay(viser_server, every_n=int(img_cfg.get("live_every_n", 2))))
    record_root = str(img_cfg.get("record_dir", "") or "")
    if record_root:
        sinks.append(NpzFrameRecorder(os.path.join(record_root, time.strftime("%Y%m%d_%H%M%S")),
                                      every_n=int(img_cfg.get("record_every_n", 1))))

    imagination = None
    if sinks:
        imagination = ImaginationRecorder(
            engine=controller.engine,
            digital_twin=digital_twin,
            sinks=sinks,
            max_rollouts=int(img_cfg.get("max_rollouts", 40)),
            ghost_count=int(img_cfg.get("ghost_count", 3)),
            scene_info=scene_info,
            roles_status=roles_status
        )
    return VisualizationHub(controller, digital_twin, dumper=dumper, imagination=imagination)


__all__ = ["VisualizationHub", "build_visualization_hub", "ImaginationRecorder", "ImaginationSink",
           "NpzFrameRecorder", "MpcInputDumper"]
