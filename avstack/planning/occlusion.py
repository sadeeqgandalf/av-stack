"""Occlusion reasoning: drive so that a pedestrian hidden behind a parked van cannot be hit hard.

The car cannot see behind a van, but it can see the van. For every occluder near
our path we check line of sight to the space just beyond its far corner, where a
person would step out. If that space is hidden, we hypothesise a "phantom"
pedestrian standing right there, about to walk into our path.

Worst case: the phantom appears just as our front bumper is at the distance
where it would be in our path exactly when we arrive. We notice it after
``t_r`` (perception + actuation), then brake at ``a``. It needs
``t_p = gap / v_ped`` to cover the lateral gap to our path, so our speed at the
moment it arrives, the impact speed, is at worst ``v - a * max(t_p - t_r, 0)``.
Requiring that to stay below a low-injury ``v_harm`` gives

        v_safe = v_harm + a * max(gap / v_ped - t_r, 0)

The closer we pass the occluder, the slower we must go, and moving away from
it (a lateral bias in the lane) buys speed back. Approaching a corner, the cap
relaxes with distance along a comfortable deceleration profile, so the car
slows early and smoothly instead of braking at the last moment.

If the phantom appears when we are too close for it to reach our path before
our front passes, it walks into our side: unavoidable for any driver, and not
what this bound is for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avstack.common.geometry import box_corners, ray_box_distances
from avstack.common.types import Track
from avstack.config import PlanningConfig, VehicleConfig
from avstack.map.hd_map import HDMap

OCCLUDER_FAMILIES = ("vehicle", "unknown")
COMFORT_DECEL = 1.5


@dataclass
class OcclusionRisk:
    track_id: int
    x: float                 # hidden emergence point (world)
    y: float
    s: float
    side: int                # +1: occluder on our left, -1: on our right
    gap: float               # lateral distance from the emergence point to our path edge [m]
    occluded: bool
    v_safe: float            # speed cap when passing the corner
    v_cap: float             # cap now, given the distance still available to slow down


class OcclusionReasoner:
    def __init__(self, cfg: PlanningConfig, vcfg: VehicleConfig, hd_map: HDMap):
        self.cfg, self.vcfg, self.map = cfg, vcfg, hd_map

    def v_safe(self, gap: float) -> float:
        c = self.cfg
        t_p = max(gap, 0.0) / c.phantom_speed
        return c.harm_speed + c.phantom_brake * max(t_p - c.phantom_reaction, 0.0)

    @staticmethod
    def _is_occluder(tr: Track) -> bool:
        # tall enough to hide a person (vehicles, large unknown objects; not cones or people)
        # and slow: nobody steps out in front of a car doing 12 m/s, but they do in front of
        # parked cars and ones stopped at a crossing
        return tr.cls.family in OCCLUDER_FAMILIES and max(tr.length, tr.width) >= 2.5 and tr.speed < 3.0

    def assess(self, ego_xy: np.ndarray, ego_s: float, ego_d: float, lane_center: float,
               tracks: list[Track]) -> list[OcclusionRisk]:
        m, c, v = self.map, self.cfg, self.vcfg
        ref = m.reference
        occluders = [t for t in tracks if self._is_occluder(t)]
        boxes = {t.id: box_corners(t.x, t.y, t.yaw, t.length, t.width) for t in occluders}
        front_s = ego_s + v.length / 2
        risks: list[OcclusionRisk] = []
        for t in occluders:
            s_o, d_o = ref.project(t.x, t.y)
            s_far = s_o + t.length / 2
            if not 0.0 < s_far - front_s < c.occlusion_lookahead:
                continue
            side = 1 if d_o > lane_center else -1
            if abs(d_o - lane_center) < 0.5 * m.lane_width:
                continue                 # a lead in our own lane: following distance covers it
            if abs(d_o - lane_center) > 2.2 * m.lane_width:
                continue                 # too far to matter
            d_inner = d_o - side * t.width / 2          # occluder edge facing our path
            # where a person would step out: just past the far corner, on the hidden side
            s_e, d_e = s_far + 0.6, d_inner
            probe_s, probe_d = s_far + 0.8, d_inner + side * 1.0
            px, py = ref.frenet_to_cartesian(probe_s, probe_d)
            ex, ey = ref.frenet_to_cartesian(s_e, d_e)
            target = np.array([float(px), float(py)])
            dist = float(np.linalg.norm(target - ego_xy))
            az = np.arctan2(target[1] - ego_xy[1], target[0] - ego_xy[0])
            others = list(boxes.values())
            hit = ray_box_distances(ego_xy, np.array([az]), others, dist)[0]
            occluded = bool(np.isfinite(hit) and hit < dist - 0.3)
            path_edge = ego_d + side * v.width / 2
            gap = max(abs(d_e - path_edge) - 0.3, 0.0)
            vs = self.v_safe(gap)
            # distance left to slow down to v_safe by the time our front reaches the corner
            room = max(s_far - front_s - 2.0, 0.0)
            v_cap = float(np.sqrt(vs**2 + 2.0 * COMFORT_DECEL * room)) if occluded else np.inf
            risks.append(OcclusionRisk(t.id, float(ex), float(ey), s_e, side, gap, occluded, vs, v_cap))
        return risks
