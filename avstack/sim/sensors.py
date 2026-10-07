"""Sensor models that turn ground truth into the noisy data the stack sees."""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import body_to_world, box_corners, normalize_angle, ray_box_distances, world_to_body
from avstack.common.timing import RateGate
from avstack.common.types import (CameraFrame, GnssMsg, ImuMsg, LaneObservation, LidarScan, RadarReturn,
                                  RadarScan, RenderedObject, SensorBundle, WheelOdomMsg)
from avstack.config import SensorConfig
from avstack.perception.camera_model import PinholeCamera

OCCLUDER_MIN_HEIGHT = 1.0      # things shorter than this (cones, debris) do not hide what is behind


class SensorSuite:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.gnss_gate = RateGate(cfg.gnss_rate, 0.05)
        self.lidar_gate = RateGate(cfg.lidar_rate, 0.0)
        self.camera_gate = RateGate(cfg.camera_rate, 0.0)      # synced to LiDAR
        self.lane_gate = RateGate(cfg.lane_rate, 0.01)
        self.radar_gate = RateGate(cfg.radar_rate, 0.02)
        self.beam_angles = np.linspace(-np.pi, np.pi, cfg.lidar_beams, endpoint=False)
        self.cameras = [PinholeCamera(cc) for cc in cfg.cameras]

    def measure(self, world) -> SensorBundle:
        t, ego, c, rng = world.t, world.ego.state, self.cfg, self.rng
        out = SensorBundle(t=t)
        out.imu = ImuMsg(t, ego.yaw_rate + c.imu_yaw_rate_bias + rng.normal(0, c.imu_yaw_rate_std),
                         ego.a + rng.normal(0, c.imu_accel_std))
        out.odom = WheelOdomMsg(t, max(0.0, ego.v + rng.normal(0, c.odom_std)))

        if self.gnss_gate.ready(t):
            nx, ny = rng.normal(0, c.gnss_std, 2)
            if rng.random() < c.gnss_outlier_prob:                     # multipath jump
                nx, ny = rng.normal(0, 8.0, 2)
            out.gnss = GnssMsg(t, ego.x + nx, ego.y + ny, c.gnss_std)

        if self.lane_gate.ready(t) and rng.random() > c.lane_dropout:
            ref = world.map.reference
            s, d = ref.project(ego.x, ego.y)
            lane = world.map.lane_at(d)
            ryaw = float(ref.interpolate(s)[2])
            out.lane = LaneObservation(
                t, d - world.map.lane_center_d(lane) + rng.normal(0, c.lane_offset_std),
                float(normalize_angle(ego.yaw - ryaw)) + rng.normal(0, c.lane_heading_std),
                c.lane_offset_std, c.lane_heading_std)

        boxes = [box_corners(a.x, a.y, a.yaw, a.length, a.width) for a in world.actors]
        if self.lidar_gate.ready(t):
            # single scan plane: objects lower than it (tyre debris, potholes) return nothing
            tall = [b for b, a in zip(boxes, world.actors) if a.height >= c.lidar_min_object_height]
            out.lidar = self._lidar(t, ego, tall)
        if self.radar_gate.ready(t):
            out.radar = self._radar(t, ego, world.actors, boxes)
        if self.camera_gate.ready(t):
            out.cameras = [self._render(t, ego, cam, world.actors, boxes) for cam in self.cameras]
        return out

    # ------------------------------------------------------------------ #
    def _lidar(self, t, ego, boxes) -> LidarScan:
        c, rng = self.cfg, self.rng
        origin = np.array([ego.x, ego.y])
        r = ray_box_distances(origin, self.beam_angles + ego.yaw, boxes, c.lidar_range)
        hit = np.isfinite(r) & (rng.random(len(r)) > c.lidar_dropout)
        r = r[hit] + rng.normal(0, c.lidar_std, hit.sum())
        ang = self.beam_angles[hit]
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        return LidarScan(t, pts)

    def _radar(self, t, ego, actors, boxes) -> RadarScan:
        """Doppler radar: one return per resolvable object, strongest (nearest) point.

        Low radar-cross-section objects (cones, debris, often pedestrians) are
        frequently missed; ghost returns appear from road furniture."""
        c, rng = self.cfg, self.rng
        mount = body_to_world(np.array([[c.radar_mount_x, 0.0]]), ego.x, ego.y, ego.yaw)[0]
        ego_vel = ego.v * np.array([np.cos(ego.yaw), np.sin(ego.yaw)])
        returns = []
        tall = [(j, b) for j, (b, a) in enumerate(zip(boxes, actors)) if a.height >= OCCLUDER_MIN_HEIGHT]
        for i, a in enumerate(actors):
            rel = np.array([a.x, a.y]) - mount
            rng_ = float(np.hypot(*rel))
            az = float(normalize_angle(np.arctan2(rel[1], rel[0]) - ego.yaw))
            in_long = abs(az) < c.radar_long_fov / 2 and rng_ < c.radar_long_range
            in_mid = abs(az) < c.radar_mid_fov / 2 and rng_ < c.radar_mid_range
            if not (in_long or in_mid):
                continue
            p_miss = c.radar_miss_prob + (0.6 if a.height < 1.0 or a.width < 1.0 else 0.0)
            if rng.random() < p_miss:
                continue
            others = [b for j, b in tall if j != i]
            first = ray_box_distances(mount, np.array([az + ego.yaw]), others, rng_)[0]
            if np.isfinite(first) and first < rng_ - a.length / 2:
                continue                                          # occluded
            u = rel / rng_
            v_obj = a.v * np.array([np.cos(a.yaw), np.sin(a.yaw)])
            if a.motion == "crossing":
                v_obj = a.d_dot * np.array([np.cos(a.yaw), np.sin(a.yaw)]) * np.sign(a.cross_speed)
            rr = float((v_obj - ego_vel) @ u)
            # the strongest reflector is the near face, not the centre
            near = rng_ - 0.5 * min(a.length, a.width)
            returns.append(RadarReturn(near + rng.normal(0, c.radar_range_std),
                                       az + rng.normal(0, c.radar_azimuth_std),
                                       rr + rng.normal(0, c.radar_range_rate_std)))
        for _ in range(rng.poisson(c.radar_clutter_rate)):
            r = rng.uniform(5, c.radar_mid_range)
            returns.append(RadarReturn(r, rng.uniform(-c.radar_mid_fov / 2, c.radar_mid_fov / 2),
                                       -ego.v * rng.uniform(0.9, 1.0) + rng.normal(0, 0.2)))  # static ghosts
        return RadarScan(t, returns)

    def _render(self, t, ego, cam: PinholeCamera, actors, boxes) -> CameraFrame:
        """Project every object's 3D box into the image, with occlusion.

        The output is what an ideal segmenter would see; the perception
        backend (SAM 3 emulator) adds the realistic failure modes on top."""
        cc = cam.cfg
        frame = CameraFrame(t, cc.name, cc.width, cc.height)
        cam_world = body_to_world(np.array([[cc.mount_x, 0.0]]), ego.x, ego.y, ego.yaw)[0]
        occluders = [(i, b) for i, (b, a) in enumerate(zip(boxes, actors)) if a.height >= OCCLUDER_MIN_HEIGHT]
        for i, a in enumerate(actors):
            if np.hypot(a.x - cam_world[0], a.y - cam_world[1]) > self.cfg.camera_range:
                continue
            fp = world_to_body(boxes[i], ego.x, ego.y, ego.yaw)
            pts = np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, a.height)]])
            u, v, depth = cam.project(pts)
            front = depth > 0.5
            if front.sum() < 2:
                continue
            u0, u1 = np.clip([u[front].min(), u[front].max()], 0, cc.width)
            v0, v1 = np.clip([v[front].min(), v[front].max()], 0, cc.height)
            if u1 - u0 < 2 or v1 - v0 < 2:
                continue
            # occlusion: cast rays through sampled columns of the object's image footprint
            az = ego.yaw + cam.azimuth_of_u(np.linspace(u0 + 1, u1 - 1, 7))
            own = ray_box_distances(cam_world, az, [boxes[i]], self.cfg.camera_range)
            others = [b for j, b in occluders if j != i]
            block = ray_box_distances(cam_world, az, others, self.cfg.camera_range)
            vis = float(np.mean(np.isfinite(own) & (own <= block + 0.05)))
            if vis < 0.25:
                continue
            frame.objects.append(RenderedObject(a.description, (float(u0), float(v0), float(u1), float(v1)), vis))
        return frame
