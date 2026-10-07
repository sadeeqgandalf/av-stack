"""A lightweight HD map: a multi-lane road built around a Frenet reference line.

Lane 0 is the right-most lane; its centre line *is* the reference line (d = 0).
Lane i has centre d = i * lane_width. Every module that reasons about lanes
works in this single Frenet frame, which keeps prediction and planning simple.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avstack.common.geometry import ReferenceLine


@dataclass
class LaneClosure:
    lane: int
    s_start: float
    s_end: float


@dataclass
class Crosswalk:
    """A marked pedestrian crossing spanning the whole road at arc length ``s``."""
    s: float
    width: float = 4.0

    @property
    def s_start(self) -> float:
        return self.s - self.width / 2

    @property
    def s_end(self) -> float:
        return self.s + self.width / 2


@dataclass
class HDMap:
    reference: ReferenceLine
    num_lanes: int = 3
    lane_width: float = 3.7
    speed_limit: float = 25.0
    closures: list[LaneClosure] = field(default_factory=list)
    crosswalks: list[Crosswalk] = field(default_factory=list)

    def crosswalks_ahead(self, s: float, horizon: float) -> list[Crosswalk]:
        return sorted((c for c in self.crosswalks if -c.width < c.s - s < horizon), key=lambda c: c.s)

    # -- lane geometry ------------------------------------------------------ #
    def lane_center_d(self, lane: int) -> float:
        return lane * self.lane_width

    def lane_at(self, d: float) -> int:
        """Lane index for a lateral offset (clamped to the road)."""
        return int(np.clip(np.round(d / self.lane_width), 0, self.num_lanes - 1))

    @property
    def road_d_min(self) -> float:
        return -self.lane_width / 2.0

    @property
    def road_d_max(self) -> float:
        return (self.num_lanes - 0.5) * self.lane_width

    def on_road(self, d: float, margin: float = 0.0) -> bool:
        return self.road_d_min - margin <= d <= self.road_d_max + margin

    def is_closed(self, lane: int, s: float) -> bool:
        return any(c.lane == lane and c.s_start <= s <= c.s_end for c in self.closures)

    def free_distance(self, lane: int, s: float, horizon: float = 1e9) -> float:
        """Distance ahead of ``s`` before ``lane`` is closed (``horizon`` if never)."""
        best = horizon
        for c in self.closures:
            if c.lane == lane and c.s_end >= s:
                best = min(best, max(c.s_start - s, 0.0))
        return best

    def lane_polyline(self, d: float, ds: float = 1.0) -> np.ndarray:
        s = np.arange(0.0, self.reference.length, ds)
        x, y = self.reference.frenet_to_cartesian(s, np.full_like(s, d))
        return np.stack([x, y], axis=1)

    def boundaries(self) -> list[tuple[str, np.ndarray]]:
        """(style, polyline) pairs for drawing: solid road edges, dashed dividers."""
        out = [("solid", self.lane_polyline(self.road_d_min)),
               ("solid", self.lane_polyline(self.road_d_max))]
        for i in range(1, self.num_lanes):
            out.append(("dashed", self.lane_polyline((i - 0.5) * self.lane_width)))
        return out


# --------------------------------------------------------------------------- #
# Road shapes used by the scenarios
# --------------------------------------------------------------------------- #
def make_road(kind: str = "curvy", length: float = 900.0) -> ReferenceLine:
    s = np.arange(0.0, length, 1.0)
    if kind == "straight":
        pts = np.stack([s, np.zeros_like(s)], axis=1)
    elif kind == "curvy":
        # gentle S-bends: heading is a smooth function of s
        yaw = 0.35 * np.sin(2 * np.pi * s / 600.0) * np.clip(s / 150.0, 0, 1)
        pts = np.stack([np.cumsum(np.cos(yaw)), np.cumsum(np.sin(yaw))], axis=1)
    else:
        raise ValueError(f"unknown road kind {kind!r}")
    return ReferenceLine(pts, resample_ds=0.5)
