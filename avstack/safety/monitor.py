"""Safety supervisor: a last line of defence, independent of the planner.

  * AEB: time-to-collision on objects inside the ego's forward corridor.
    Triggers full braking below ``aeb_ttc``, releases above ``aeb_release_ttc``
    (hysteresis avoids brake chatter).
  * Watchdog: if the planner has not produced a fresh trajectory recently,
    command a controlled stop.

Deliberately simple and conservative: it reasons directly on tracks, not
on predictions, so a prediction or planning bug cannot disable it.
"""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import world_to_body
from avstack.common.types import ControlCommand, Track, VehicleState
from avstack.config import SafetyConfig, VehicleConfig


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig, vcfg: VehicleConfig):
        self.cfg, self.vcfg = cfg, vcfg
        self.aeb_active = False
        self.min_ttc = np.inf
        self.aeb_events = 0

    def time_to_collision(self, ego: VehicleState, tracks: list[Track]) -> float:
        front = self.vcfg.rear_to_center + self.vcfg.length / 2
        best = np.inf
        for tr in tracks:
            # Phantom-braking guard: brake only for objects corroborated by two sensing
            # modalities (or tracked for a long time) with physically plausible motion.
            corroborated = len(tr.sources) >= 2 or tr.hits >= 30
            if not corroborated or tr.speed > 60.0:
                continue
            rel = world_to_body(np.array([[tr.x, tr.y]]), ego.x, ego.y, ego.yaw)[0]
            if rel[0] <= 0 or abs(rel[1]) > self.cfg.corridor_half_width + tr.width / 2:
                continue
            gap = rel[0] - front - tr.length / 2
            v_obj = tr.vx * np.cos(ego.yaw) + tr.vy * np.sin(ego.yaw)
            closing = ego.v - v_obj
            if gap <= 0.0:
                return 0.0
            if closing > 0.1:
                best = min(best, gap / closing)
        return best

    def check(self, t: float, ego: VehicleState, tracks: list[Track], cmd: ControlCommand,
              plan_time: float | None) -> ControlCommand:
        c = self.cfg
        self.min_ttc = self.time_to_collision(ego, tracks)
        if not self.aeb_active and self.min_ttc < c.aeb_ttc and ego.v > 0.5:
            self.aeb_active = True
            self.aeb_events += 1
        elif self.aeb_active and (self.min_ttc > c.aeb_release_ttc or ego.v < 0.1):
            self.aeb_active = False

        if self.aeb_active:
            return ControlCommand(accel=-self.vcfg.max_decel, steer=cmd.steer, source="aeb")
        if plan_time is None or t - plan_time > c.max_plan_age:
            return ControlCommand(accel=-3.0, steer=cmd.steer, source="fallback")
        return cmd
