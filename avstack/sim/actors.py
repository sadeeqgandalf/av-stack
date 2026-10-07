"""Background traffic: IDM car-following in Frenet space with scripted lane changes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avstack.common.types import ObjectClass


@dataclass
class LaneChangeTrigger:
    """Fire a lane change when the time or ego-relative condition is met."""
    to_lane: int
    at_time: float | None = None
    when_ego_behind: float | None = None   # fire when ego is this close behind in s
    fired: bool = False


@dataclass
class Actor:
    """Any road user or obstacle.

    motion:
      "idm"       lane-following vehicle / cyclist (IDM car-following, lane changes)
      "static"    never moves (cone, barrier, debris, parked or broken-down car)
      "crossing"  walks across the road (d changes) starting at ``cross_start``
    """
    id: int
    s: float
    lane: int
    v: float
    desired_v: float
    length: float = 4.6
    width: float = 1.9
    cls: ObjectClass = ObjectClass.CAR
    d: float | None = None                 # defaults to lane centre (+ lane_offset)
    triggers: list[LaneChangeTrigger] = field(default_factory=list)
    lat_speed: float = 1.2                 # m/s lateral during lane change
    height: float = 1.5                    # [m] objects below the LiDAR plane are invisible to it
    description: str = "car"               # appearance, as the camera would describe it
    motion: str = "idm"
    lane_offset: float = 0.0               # e.g. cyclists ride near the right edge
    cross_start: float = 0.0
    cross_when_ego_within: float | None = None   # step out when the ego is this close (jaywalker)
    cross_speed: float = 1.4               # signed d_dot while crossing
    cross_to_d: float = 0.0
    # filled by the world each step
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    d_dot: float = 0.0
    a: float = 0.0


def idm_accel(v: float, v0: float, gap: float, dv: float,
              a_max: float = 1.5, b: float = 2.5, s0: float = 3.0, T: float = 1.4) -> float:
    """Intelligent Driver Model (Treiber et al.)."""
    if v0 <= 0.1:
        return -b if v > 0 else 0.0
    free = 1.0 - (v / v0) ** 4
    if not np.isfinite(gap):
        return a_max * free
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * np.sqrt(a_max * b)))
    return float(np.clip(a_max * (free - (s_star / max(gap, 0.1)) ** 2), -8.0, a_max))
