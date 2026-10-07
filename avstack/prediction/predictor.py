"""Motion prediction for tracked objects.

Lane-following model in the map's Frenet frame:
  * longitudinal: constant speed along the lane (s_dot held),
  * lateral: exponential convergence to a target lane centre. If the object
    already has significant lateral velocity, the target is the *adjacent*
    lane it is moving toward, which is how cut-ins get predicted early.
Objects that do not fit the map (off-road, wrong-way) fall back to constant
velocity in Cartesian space. Uncertainty grows linearly with horizon and is
consumed by the planner as extra collision margin.
"""
from __future__ import annotations

import numpy as np

from avstack.common.types import ObjectClass, PredictedObject, Track
from avstack.config import PredictionConfig
from avstack.map.hd_map import HDMap


class LanePredictor:
    def __init__(self, cfg: PredictionConfig, hd_map: HDMap, lane_change_d_dot: float = 0.5):
        self.cfg = cfg
        self.map = hd_map
        self.lc_d_dot = lane_change_d_dot
        self.ts = np.arange(0.0, cfg.horizon + 1e-9, cfg.dt)

    def predict(self, tracks: list[Track]) -> list[PredictedObject]:
        return [self._predict_one(tr) for tr in tracks]

    def _predict_one(self, tr: Track) -> PredictedObject:
        fam = tr.cls.family
        # below 0.5 m/s a pedestrian's tracked velocity is mostly tracker jitter: treat as standing
        if fam == "static" or (fam == "vru" and not tr.moving) or (fam == "unknown" and not tr.moving
                                                                     and tr.length < 3.0):
            return self._static(tr)
        if fam == "vehicle" and tr.speed < 0.5:
            return self._static(tr)          # parked / stopped: it will not drift to a lane centre
        if fam == "vru":
            # cyclists hold their line far better than pedestrians or animals
            lat = 0.2 if tr.cls == ObjectClass.BICYCLE else 0.5
            return self._constant_velocity(tr, lat_growth=lat, lon_growth=0.6)
        return self._lane_following(tr)

    def _static(self, tr: Track) -> PredictedObject:
        ts = self.ts
        # a standing person may start walking, but a 5 s isotropic blow-up turns someone on the
        # far pavement into a wall across the lane. Grow modestly and cap it: real intent to
        # cross (someone at a kerb by a crossing) is handled explicitly by the crosswalk rules.
        vru = tr.cls.family in ("vru", "unknown")
        std = np.minimum(0.2 + (0.2 if vru else 0.02) * ts, 1.0 if vru else 0.3)
        return PredictedObject(tr.id, ts, np.full_like(ts, tr.x), np.full_like(ts, tr.y), np.full_like(ts, tr.yaw),
                               tr.length, tr.width, std, std.copy(), tr.cls, "static")

    def _constant_velocity(self, tr: Track, lat_growth: float, lon_growth: float) -> PredictedObject:
        """VRUs do not follow lanes: pedestrians cross, cyclists swerve."""
        ts = self.ts
        x, y = tr.x + tr.vx * ts, tr.y + tr.vy * ts
        heading = np.arctan2(tr.vy, tr.vx) if tr.speed > 0.3 else tr.yaw
        # beyond ~1.5 m, lateral 1-sigma stops being information and just freezes the planner
        return PredictedObject(tr.id, ts, x, y, np.full_like(ts, heading), tr.length, tr.width,
                               0.3 + lon_growth * ts, np.minimum(0.3 + lat_growth * ts, 1.5), tr.cls, "cv")

    def _lane_following(self, tr: Track) -> PredictedObject:
        ref, ts = self.map.reference, self.ts
        speed = tr.speed
        heading = np.arctan2(tr.vy, tr.vx) if speed > 1.0 else tr.yaw
        s0, d0, s_dot, d_dot = ref.cartesian_to_frenet_state(tr.x, tr.y, heading, speed)
        c = self.cfg
        lon_std = 0.3 + c.lon_std_growth * ts

        if not self.map.on_road(d0, 1.0) or s_dot < -0.5:
            x = tr.x + tr.vx * ts
            y = tr.y + tr.vy * ts
            yaw = np.full_like(ts, heading)
            lat_std = 0.3 + c.lat_std_growth_maneuver * ts
            return PredictedObject(tr.id, ts, x, y, yaw, tr.length, tr.width, lon_std, lat_std, tr.cls, "cv")

        d_probe = d0
        maneuvering = abs(d_dot) > self.lc_d_dot
        lat_std = 0.2 + (c.lat_std_growth_maneuver if maneuvering else c.lat_std_growth) * ts
        if maneuvering:
            # lane change in progress: which lane will the lateral motion land in?
            d_probe = d0 + np.clip(d_dot * 2.0, -self.map.lane_width, self.map.lane_width)
        lane_c = self.map.lane_center_d(self.map.lane_at(d_probe))
        # a vehicle riding the shoulder or straddling the edge keeps its line; only in-lane
        # vehicles (or ones clearly changing lanes) converge to a lane centre
        d_tgt = lane_c if (maneuvering or abs(d0 - lane_c) < 0.35 * self.map.lane_width) else d0
        tau = self.cfg.lateral_convergence_time / 3.0
        d = d_tgt + (d0 - d_tgt) * np.exp(-ts / tau)
        s = s0 + max(s_dot, 0.0) * ts
        x, y = ref.frenet_to_cartesian(s, d)
        if len(ts) > 1:
            yaw = np.arctan2(np.gradient(y), np.gradient(x)) if speed > 1.0 else np.full_like(ts, tr.yaw)
        else:
            yaw = np.array([heading])
        return PredictedObject(tr.id, ts, np.asarray(x), np.asarray(y), yaw, tr.length, tr.width,
                               lon_std, lat_std, tr.cls, "lane")
