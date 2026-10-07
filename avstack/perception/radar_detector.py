"""Radar returns -> world-frame detections carrying Doppler range rate.

Radar is the long-range velocity sensor: position is coarse laterally
(range x azimuth noise) but range rate is measured directly, so a stalled
car at 150 m is known to be stationary long before LiDAR can see it.
"""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import body_to_world
from avstack.common.types import Detection, ObjectClass, RadarScan, VehicleState
from avstack.config import PerceptionConfig, SensorConfig
from avstack.map.hd_map import HDMap

NEAR_FACE_TO_CENTRE = 1.2      # radar ranges the near face; typical vehicle half-length-ish


class RadarDetector:
    def __init__(self, cfg: PerceptionConfig, sensor_cfg: SensorConfig, hd_map: HDMap):
        self.cfg, self.scfg, self.map = cfg, sensor_cfg, hd_map

    def detect(self, scan: RadarScan, ego: VehicleState) -> list[Detection]:
        c = self.scfg
        mount = body_to_world(np.array([[c.radar_mount_x, 0.0]]), ego.x, ego.y, ego.yaw)[0]
        ego_vel = (ego.v * np.cos(ego.yaw), ego.v * np.sin(ego.yaw))
        out = []
        for r in scan.returns:
            bearing = ego.yaw + r.azimuth
            u = np.array([np.cos(bearing), np.sin(bearing)])
            centre = mount + u * (r.range + NEAR_FACE_TO_CENTRE)
            _, d = self.map.reference.project(*centre)
            if not self.map.on_road(d, self.cfg.roi_margin):
                continue
            lateral = r.range * c.radar_azimuth_std
            out.append(Detection(x=float(centre[0]), y=float(centre[1]), yaw=float(bearing),
                                 length=4.5, width=1.9, cls=ObjectClass.UNKNOWN, source="radar",
                                 pos_std=float(max(0.6, np.hypot(1.0, 2.0 * lateral))),
                                 range_rate=r.range_rate, sensor_xy=(float(mount[0]), float(mount[1])),
                                 sensor_vel=(float(ego_vel[0]), float(ego_vel[1]))))
        return out
