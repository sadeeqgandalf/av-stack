"""Geometry primitives shared across the stack.

Conventions (used everywhere in the stack):
  * World frame is ENU-like, x east, y north, yaw counter-clockwise from +x.
  * Frenet frame: s is arc length along a reference line, d is lateral offset,
    positive to the LEFT of the direction of travel.
"""
from __future__ import annotations

import numpy as np


def normalize_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def world_to_body(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Transform Nx2 world points into a body frame located at (x, y, yaw)."""
    return (np.asarray(points) - np.array([x, y])) @ rot2d(yaw)


def body_to_world(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Transform Nx2 body-frame points into the world frame."""
    return np.asarray(points) @ rot2d(yaw).T + np.array([x, y])


# --------------------------------------------------------------------------- #
# Oriented boxes
# --------------------------------------------------------------------------- #
def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Return 4x2 corners (counter-clockwise) of an oriented rectangle."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return body_to_world(local, x, y, yaw)


def _project_polygon(corners: np.ndarray, axis: np.ndarray):
    p = corners @ axis
    return p.min(), p.max()


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating Axis Theorem test for two convex polygons (Nx2, Mx2)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        for n in normals:
            amin, amax = _project_polygon(a, n)
            bmin, bmax = _project_polygon(b, n)
            if amax < bmin or bmax < amin:
                return False
    return True


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between two convex polygons (0 if they overlap)."""
    if polygons_overlap(a, b):
        return 0.0
    best = np.inf
    for P, Q in ((a, b), (b, a)):
        for p in P:
            for i in range(len(Q)):
                best = min(best, _point_segment_dist(p, Q[i], Q[(i + 1) % len(Q)]))
    return best


def ray_box_distances(origin: np.ndarray, angles: np.ndarray, boxes: list[np.ndarray],
                      max_range: float) -> np.ndarray:
    """Cast rays from ``origin`` at world ``angles`` against box polygons.

    Returns the hit distance per ray (``inf`` when nothing is hit within range).
    Fully vectorised: rays x edges.
    """
    ranges = np.full(len(angles), np.inf)
    if not boxes:
        return ranges
    edges_a = np.concatenate([b for b in boxes])                                   # (E,2)
    edges_b = np.concatenate([np.roll(b, -1, axis=0) for b in boxes])              # (E,2)
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)                         # (R,2)
    e = edges_b - edges_a                                                          # (E,2)
    w = edges_a - origin                                                           # (E,2)
    # Solve origin + t*d = a + u*e  ->  cross products
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]          # (R,E)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
        u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / denom
    valid = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1) & (t <= max_range)
    t = np.where(valid, t, np.inf)
    return np.minimum(ranges, t.min(axis=1))


# --------------------------------------------------------------------------- #
# Reference line (the backbone of the map, prediction and Frenet planning)
# --------------------------------------------------------------------------- #
class ReferenceLine:
    """A smooth, arc-length parameterised 2D curve with Frenet conversions."""

    def __init__(self, points: np.ndarray, resample_ds: float = 0.5):
        pts = np.asarray(points, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-6])
        pts = pts[keep]
        s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])

        n = max(int(np.ceil(s_raw[-1] / resample_ds)) + 1, 2)
        self.s = np.linspace(0.0, s_raw[-1], n)
        self.x = np.interp(self.s, s_raw, pts[:, 0])
        self.y = np.interp(self.s, s_raw, pts[:, 1])

        dx, dy = np.gradient(self.x, self.s), np.gradient(self.y, self.s)
        self.yaw = np.unwrap(np.arctan2(dy, dx))
        kappa = np.gradient(self.yaw, self.s)
        # light smoothing: curvature from resampled polylines is noisy
        k = 7
        self.kappa = np.convolve(np.pad(kappa, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        self.length = float(self.s[-1])
        self._xy = np.stack([self.x, self.y], axis=1)

    # -- queries ------------------------------------------------------------ #
    def interpolate(self, s):
        """Return (x, y, yaw, kappa) at arc length(s) ``s`` (clipped to the line)."""
        s = np.clip(s, 0.0, self.length)
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        yaw = normalize_angle(np.interp(s, self.s, self.yaw))
        kappa = np.interp(s, self.s, self.kappa)
        return x, y, yaw, kappa

    def frenet_to_cartesian(self, s, d):
        x, y, yaw, _ = self.interpolate(s)
        return x - np.asarray(d) * np.sin(yaw), y + np.asarray(d) * np.cos(yaw)

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Project a world point onto the line -> (s, d)."""
        p = np.array([x, y])
        a, b = self._xy[:-1], self._xy[1:]
        ab = b - a
        L2 = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist2 = np.einsum("ij,ij->i", p - proj, p - proj)
        i = int(np.argmin(dist2))
        s = self.s[i] + t[i] * np.sqrt(L2[i])
        cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
        d = np.sign(cross) * np.sqrt(dist2[i])
        return float(s), float(d)

    def cartesian_to_frenet_state(self, x, y, yaw, v):
        """Project a full kinematic state -> (s, d, s_dot, d_dot)."""
        s, d = self.project(x, y)
        _, _, ryaw, kappa = self.interpolate(s)
        dyaw = normalize_angle(yaw - ryaw)
        s_dot = v * np.cos(dyaw) / max(1.0 - kappa * d, 1e-3)
        d_dot = v * np.sin(dyaw)
        return s, d, float(s_dot), float(d_dot)
