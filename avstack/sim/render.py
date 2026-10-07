"""Per-camera ground-truth rendering: depth, semantic, instance and a shaded RGB proxy.

A vectorised ray caster over the simulator's 3D world (flat ground, road from the HD map,
every actor as a box: footprint x height), using exactly the calibration the stack uses
(``PinholeCamera``). Per pixel it returns

  depth     planar depth along the camera's forward axis [m] (what depth maps store)
  semantic  CARLA semantic tag (Cityscapes-style, CARLA >= 0.9.14), so logs from this
            simulator and from CARLA share one label space
  instance  actor id + 1 (0 = no actor)
  rgb       a simple shaded rendering (sky, asphalt, markings, class-coloured boxes)

Honest scope: geometry is boxes on a plane. Depth / semantic / instance are exact and
ideal as *control inputs* (Cosmos-Transfer) and as *ground truth* for scoring perception;
the RGB proxy is not photoreal: people are 0.6 m boxes. Photoreal pixels come from CARLA
or from Cosmos run on these control maps.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from avstack.common.types import ObjectClass
from avstack.config import CameraConfig
from avstack.map.hd_map import HDMap

# CARLA semantic tags (0.9.14+). Verify against your CARLA build's documentation.
CARLA_TAGS = {
    "unlabeled": 0, "road": 1, "sidewalk": 2, "building": 3, "wall": 4, "fence": 5, "pole": 6,
    "traffic_light": 7, "traffic_sign": 8, "vegetation": 9, "terrain": 10, "sky": 11, "pedestrian": 12,
    "rider": 13, "car": 14, "truck": 15, "bus": 16, "train": 17, "motorcycle": 18, "bicycle": 19,
    "static": 20, "dynamic": 21, "other": 22, "water": 23, "roadline": 24, "ground": 25,
}
CLASS_TO_TAG = {
    ObjectClass.CAR: "car", ObjectClass.TRUCK: "truck", ObjectClass.BUS: "bus",
    ObjectClass.MOTORCYCLE: "motorcycle", ObjectClass.BICYCLE: "bicycle", ObjectClass.PEDESTRIAN: "pedestrian",
    ObjectClass.ANIMAL: "dynamic", ObjectClass.TRAFFIC_CONE: "static", ObjectClass.BARRIER: "static",
    ObjectClass.DEBRIS: "other", ObjectClass.UNKNOWN: "other",
}
# display palette (CARLA / Cityscapes colours) for colourised semantic videos
TAG_COLORS = {
    0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232), 9: (107, 142, 35), 10: (152, 251, 152),
    11: (70, 130, 180), 12: (220, 20, 60), 13: (255, 0, 0), 14: (0, 0, 142), 15: (0, 0, 70), 16: (0, 60, 100),
    18: (0, 0, 230), 19: (119, 11, 32), 20: (110, 190, 160), 21: (170, 120, 50), 22: (55, 90, 80),
    24: (157, 234, 50), 25: (81, 0, 81),
}
# proxy RGB base colours per object class (muted, car-paint-like)
OBJ_RGB = {
    ObjectClass.CAR: (70, 90, 130), ObjectClass.TRUCK: (170, 170, 165), ObjectClass.BUS: (190, 160, 60),
    ObjectClass.MOTORCYCLE: (60, 60, 70), ObjectClass.BICYCLE: (150, 40, 40), ObjectClass.PEDESTRIAN: (120, 80, 60),
    ObjectClass.ANIMAL: (140, 100, 60), ObjectClass.TRAFFIC_CONE: (240, 110, 20), ObjectClass.BARRIER: (230, 230, 225),
    ObjectClass.DEBRIS: (40, 40, 40), ObjectClass.UNKNOWN: (120, 120, 120),
}
MAX_RANGE = 200.0
SIDEWALK_WIDTH = 3.0


class CameraRenderer:
    def __init__(self, cam: CameraConfig, hd_map: HDMap, width: int | None = None, height: int | None = None):
        self.cfg = cam
        self.map = hd_map
        self.W = width or cam.width
        self.H = height or cam.height
        # intrinsics at the render resolution (same hfov as the stack's camera)
        self.fx = self.W / 2.0 / np.tan(cam.hfov / 2.0)
        self.fy = self.fx
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        # camera-frame ray per pixel, forward component = 1 so t is planar depth
        self.left = (self.cx - u) / self.fx
        self.up = (self.cy - v) / self.fy
        ref = hd_map.reference
        self._ref_xy = np.stack([ref.x, ref.y], axis=1)
        self._ref_tree = cKDTree(self._ref_xy)
        self._ref_yaw = ref.yaw
        self._ref_s = ref.s

    @property
    def intrinsics(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": self.W, "height": self.H}

    def _frenet(self, x: np.ndarray, y: np.ndarray):
        """Approximate (s, d) for many world points via the nearest reference sample."""
        _, i = self._ref_tree.query(np.stack([x, y], axis=1), workers=-1)
        dx, dy = x - self._ref_xy[i, 0], y - self._ref_xy[i, 1]
        yaw = self._ref_yaw[i]
        return self._ref_s[i] + dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)

    def render(self, ego_x: float, ego_y: float, ego_yaw: float, actors) -> dict:
        c = self.cfg
        H, W = self.H, self.W
        psi = ego_yaw + c.yaw
        ox = ego_x + c.mount_x * np.cos(ego_yaw)
        oy = ego_y + c.mount_x * np.sin(ego_yaw)
        oz = c.mount_z
        dx = np.cos(psi) - self.left * np.sin(psi)
        dy = np.sin(psi) + self.left * np.cos(psi)
        dz = self.up

        depth = np.full((H, W), np.inf)
        semantic = np.full((H, W), CARLA_TAGS["sky"], dtype=np.uint8)
        instance = np.zeros((H, W), dtype=np.uint16)
        face = np.zeros((H, W), dtype=np.int8)                 # which box face was hit (for shading)
        obj_cls = np.full((H, W), -1, dtype=np.int16)

        # ---- ground plane ---------------------------------------------------
        down = dz < -1e-6
        tg = np.where(down, oz / np.where(down, -dz, 1.0), np.inf)
        tg[tg > MAX_RANGE] = np.inf
        ground = np.isfinite(tg)
        depth[ground] = tg[ground]

        # ---- actors (boxes): slab test in each box's own frame -------------
        classes = list(ObjectClass)
        for a in actors:
            if np.hypot(a.x - ox, a.y - oy) > MAX_RANGE:
                continue
            ca, sa = np.cos(a.yaw), np.sin(a.yaw)
            lx0 = (ox - a.x) * ca + (oy - a.y) * sa                 # ray origin in box frame
            ly0 = -(ox - a.x) * sa + (oy - a.y) * ca
            ldx = dx * ca + dy * sa
            ldy = -dx * sa + dy * ca
            hl, hw, h = a.length / 2, a.width / 2, max(a.height, 0.05)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1x, t2x = (-hl - lx0) / ldx, (hl - lx0) / ldx
                t1y, t2y = (-hw - ly0) / ldy, (hw - ly0) / ldy
                t1z, t2z = (0.0 - oz) / dz, (h - oz) / dz
            tminx, tmaxx = np.minimum(t1x, t2x), np.maximum(t1x, t2x)
            tminy, tmaxy = np.minimum(t1y, t2y), np.maximum(t1y, t2y)
            tminz, tmaxz = np.minimum(t1z, t2z), np.maximum(t1z, t2z)
            tnear = np.maximum(np.maximum(tminx, tminy), tminz)
            tfar = np.minimum(np.minimum(tmaxx, tmaxy), tmaxz)
            hit = (tnear <= tfar) & (tnear > 0.1) & (tnear < depth)
            if not hit.any():
                continue
            depth[hit] = tnear[hit]
            instance[hit] = a.id + 1
            semantic[hit] = CARLA_TAGS[CLASS_TO_TAG[a.cls]]
            obj_cls[hit] = classes.index(a.cls)
            # face: 1 = end (+/-x), 2 = side (+/-y), 3 = top
            which = np.where(tnear == tminz, 3, np.where(tnear == tminx, 1, 2))
            face[hit] = which[hit]

        # ---- label ground pixels by the HD map ------------------------------
        g = ground & (instance == 0)
        gx, gy = ox + dx[g] * tg[g], oy + dy[g] * tg[g]
        s, d = self._frenet(gx, gy)
        m = self.map
        on_road = (d >= m.road_d_min) & (d <= m.road_d_max)
        sidewalk = ~on_road & (d >= m.road_d_min - SIDEWALK_WIDTH) & (d <= m.road_d_max + SIDEWALK_WIDTH)
        tags = np.where(on_road, CARLA_TAGS["road"], np.where(sidewalk, CARLA_TAGS["sidewalk"], CARLA_TAGS["terrain"]))
        line = np.zeros_like(on_road)
        for edge in (m.road_d_min, m.road_d_max):
            line |= np.abs(d - edge) < 0.12
        for k in range(1, m.num_lanes):
            line |= (np.abs(d - (k - 0.5) * m.lane_width) < 0.08) & ((s % 12.0) < 4.0)   # dashed dividers
        for cw in m.crosswalks:
            inside = (s >= cw.s_start) & (s <= cw.s_end) & on_road
            line |= inside & ((np.floor((d - m.road_d_min) / 0.5) % 2) == 0)          # zebra stripes
        tags = np.where(line & on_road, CARLA_TAGS["roadline"], tags)
        semantic[g] = tags

        return {"depth": depth, "semantic": semantic, "instance": instance,
                "rgb": self._shade(depth, semantic, face, obj_cls, classes)}

    def _shade(self, depth, semantic, face, obj_cls, classes) -> np.ndarray:
        """Cheap but legible proxy RGB: flat colours, face shading, distance haze."""
        H, W = semantic.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        sky_t = np.clip((np.arange(H) / max(self.cy, 1)), 0, 1)[:, None, None]
        sky = (1 - sky_t) * np.array([90, 130, 190]) + sky_t * np.array([185, 205, 225])
        rgb[:] = np.broadcast_to(sky, (H, W, 3))
        palette = {CARLA_TAGS["road"]: (72, 72, 76), CARLA_TAGS["roadline"]: (225, 225, 215),
                   CARLA_TAGS["sidewalk"]: (150, 145, 140), CARLA_TAGS["terrain"]: (90, 115, 70)}
        for tag, col in palette.items():
            rgb[semantic == tag] = col
        obj = obj_cls >= 0
        if obj.any():
            base = np.array([OBJ_RGB[cl] for cl in classes], dtype=np.float32)[obj_cls[obj]]
            shade = np.select([face[obj] == 3, face[obj] == 1], [1.15, 0.95], 0.75)[:, None]
            rgb[obj] = base * shade
        finite = np.isfinite(depth)
        haze = np.clip(np.where(finite, depth, 0) / MAX_RANGE, 0, 1)[..., None] * 0.55
        rgb = np.where(finite[..., None], rgb * (1 - haze) + np.array([185, 200, 215]) * haze, rgb)
        return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for tag, col in TAG_COLORS.items():
        lut[tag] = col
    return lut[semantic]
