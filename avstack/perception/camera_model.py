"""Pinhole camera geometry in the vehicle body frame (rear axle, x fwd, y left, z up).

Shared calibration: the simulator renders with it and the stack lifts 2D
detections back into 3D with it, exactly like a vehicle calibration file.
"""
from __future__ import annotations

import numpy as np

from avstack.config import CameraConfig


class PinholeCamera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.fx = cfg.width / 2.0 / np.tan(cfg.hfov / 2.0)
        self.fy = self.fx
        self.cx, self.cy = cfg.width / 2.0, cfg.height / 2.0

    def body_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 body points -> Nx3 camera (forward, left, up) points."""
        c, s = np.cos(self.cfg.yaw), np.sin(self.cfg.yaw)
        rel = pts - np.array([self.cfg.mount_x, 0.0, 0.0])
        fwd = rel[:, 0] * c + rel[:, 1] * s
        left = -rel[:, 0] * s + rel[:, 1] * c
        return np.stack([fwd, left, rel[:, 2]], axis=1)

    def project(self, pts_body: np.ndarray):
        """Nx3 body points -> (u, v, depth). Points behind the camera get depth <= 0."""
        cam = self.body_to_cam(pts_body)
        depth = cam[:, 0]
        safe = np.where(depth > 1e-3, depth, np.nan)
        u = self.cx - self.fx * cam[:, 1] / safe
        v = self.cy + self.fy * (self.cfg.mount_z - cam[:, 2]) / safe
        return u, v, depth

    # -- inverse models used for lifting ------------------------------------ #
    def azimuth_of_u(self, u):
        """Body-frame bearing of image column(s) u."""
        return self.cfg.yaw + np.arctan2(self.cx - np.asarray(u), self.fx)

    def ground_depth_of_v(self, v: float) -> float:
        """Flat-ground monocular depth: the bottom edge of an object touches the road."""
        dv = v - self.cy
        return float(self.fy * self.cfg.mount_z / dv) if dv > 1.0 else float("inf")

    def cam_to_body_xy(self, depth: float, u: float) -> np.ndarray:
        """Ground-plane point at forward ``depth`` along column ``u`` -> body (x, y)."""
        left = (self.cx - u) * depth / self.fx
        c, s = np.cos(self.cfg.yaw), np.sin(self.cfg.yaw)
        return np.array([self.cfg.mount_x + depth * c - left * s, depth * s + left * c])
