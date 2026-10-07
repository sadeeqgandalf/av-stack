"""LiDAR object detection: clustering + L-shape box fitting.

Pipeline:
  1. Transform points body -> world using the *estimated* ego pose.
  2. Euclidean clustering (KD-tree radius graph + connected components).
  3. Fit an oriented box per cluster with the L-shape "closeness" criterion
     (Zhang et al., "Efficient L-Shape Fitting for Vehicle Detection", 2017).
  4. Resolve the 90-degree box ambiguity with the lane heading from the map.
  5. Complete partially-seen boxes: LiDAR only sees the near faces, so a car
     seen end-on looks 2 m long. Grow short sides *away* from the sensor.
  6. Map ROI filter: drop clusters that are off the road.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from avstack.common.geometry import body_to_world, normalize_angle, rot2d
from avstack.common.types import Detection, LidarScan, ObjectClass, VehicleState
from avstack.config import PerceptionConfig
from avstack.map.hd_map import HDMap

_ANGLES = np.deg2rad(np.arange(0.0, 90.0, 1.0))


def euclidean_clusters(points: np.ndarray, tol: float, min_pts: int,
                       origin: np.ndarray | None = None, beam_spacing: float = 0.0) -> list[np.ndarray]:
    """Radius-graph clustering. The radius grows with range because beam spacing
    does: at 60 m, 0.5-degree beams land ~0.5 m apart on a surface and much further
    apart on a grazing side panel, which would otherwise split one car in two."""
    if len(points) < min_pts:
        return []
    rng = np.linalg.norm(points - origin, axis=1) if origin is not None else np.zeros(len(points))
    max_tol = tol + 3.0 * beam_spacing * (rng.max() if len(rng) else 0.0)
    pairs = cKDTree(points).query_pairs(max_tol, output_type="ndarray")
    if len(pairs):
        dist = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        allowed = tol + 3.0 * beam_spacing * np.minimum(rng[pairs[:, 0]], rng[pairs[:, 1]])
        pairs = pairs[dist <= allowed]
    else:
        pairs = np.zeros((0, 2), dtype=int)
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return [points[labels == k] for k in np.unique(labels) if (labels == k).sum() >= min_pts]


def fit_lshape(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (theta, min_xy, max_xy) of the best rectangle in the theta-rotated frame."""
    best_score, best = -np.inf, None
    for th in _ANGLES:
        e1, e2 = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        c1, c2 = points @ e1, points @ e2
        d1 = np.minimum(c1.max() - c1, c1 - c1.min())
        d2 = np.minimum(c2.max() - c2, c2 - c2.min())
        d = np.maximum(np.minimum(d1, d2), 0.01)
        score = np.sum(1.0 / d)
        if score > best_score:
            best_score = score
            best = (th, np.array([c1.min(), c2.min()]), np.array([c1.max(), c2.max()]))
    return best


class LidarDetector:
    def __init__(self, cfg: PerceptionConfig, hd_map: HDMap, beam_spacing: float = np.deg2rad(0.5)):
        self.cfg = cfg
        self.beam_spacing = beam_spacing
        self.map = hd_map
        self.last_world_points = np.zeros((0, 2))

    def detect(self, scan: LidarScan, ego: VehicleState) -> list[Detection]:
        cfg = self.cfg
        pts = body_to_world(scan.points, ego.x, ego.y, ego.yaw) if len(scan.points) else np.zeros((0, 2))
        self.last_world_points = pts
        out = []
        clusters = euclidean_clusters(pts, cfg.cluster_tolerance, cfg.min_cluster_points,
                                      origin=np.array([ego.x, ego.y]), beam_spacing=self.beam_spacing)
        for cl in clusters:
            if np.ptp(cl, axis=0).max() > cfg.max_cluster_extent:
                continue                                            # walls / guard rails
            det = self._box_from_cluster(cl, ego)
            s, d = self.map.reference.project(det.x, det.y)
            if self.map.on_road(d, cfg.roi_margin):
                out.append(det)
        return out

    def _box_from_cluster(self, cl: np.ndarray, ego: VehicleState) -> Detection:
        cfg = self.cfg
        extent = np.ptp(cl, axis=0)
        if extent.max() < cfg.small_object_extent:
            # pedestrians, cones, poles: never "complete" these into car-sized boxes
            c = cl.mean(axis=0)
            size = float(max(extent.max(), 0.4))
            return Detection(x=float(c[0]), y=float(c[1]), yaw=0.0, length=size, width=size,
                             cls=ObjectClass.UNKNOWN, source="lidar", pos_std=cfg.lidar_pos_std)
        th, lo, hi = fit_lshape(cl)
        # choose the box axis closest to the lane direction as the "length" axis
        s, _ = self.map.reference.project(*cl.mean(axis=0))
        road_yaw = float(self.map.reference.interpolate(s)[2])
        k = int(np.round(normalize_angle(road_yaw - th) / (np.pi / 2)))
        yaw = th + k * np.pi / 2
        # express the box extents in the (yaw) frame
        R = rot2d(yaw)
        local = cl @ R
        lo, hi = local.min(axis=0), local.max(axis=0)
        ego_local = np.array([ego.x, ego.y]) @ R
        target = np.array([cfg.default_car_length, cfg.default_car_width])
        for i in range(2):
            if hi[i] - lo[i] < target[i]:
                # the far side is hidden: extend away from the sensor
                if ego_local[i] <= lo[i]:
                    hi[i] = lo[i] + target[i]
                elif ego_local[i] >= hi[i]:
                    lo[i] = hi[i] - target[i]
                else:
                    mid = 0.5 * (lo[i] + hi[i])
                    lo[i], hi[i] = mid - target[i] / 2, mid + target[i] / 2
        center = R @ (0.5 * (lo + hi))
        length, width = hi - lo
        return Detection(x=float(center[0]), y=float(center[1]), yaw=float(normalize_angle(yaw)),
                         length=float(max(length, cfg.min_box_dim)), width=float(max(width, cfg.min_box_dim)),
                         cls=ObjectClass.TRUCK if length > 7.0 else ObjectClass.CAR,
                         source="lidar", pos_std=cfg.lidar_pos_std)
