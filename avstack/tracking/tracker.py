"""Multi-object tracking with track-level LiDAR/camera fusion.

  * Class-dependent motion models. Vehicles: CTRV EKF [x, y, v, yaw, yaw_rate]
    (constant-velocity models lag on curves, which reads as a phantom lateral
    velocity, i.e. a fake cut-in). Pedestrians, animals, small objects:
    holonomic constant velocity [x, y, vx, vy] (a dog can run sideways; a
    CTRV seeded with the lane heading cannot follow it and the track churns).
  * Class-aware: vehicles, vulnerable road users and static objects get
    different process noise and initial velocity priors.
  * Association: Mahalanobis distance, chi^2 gating, optimal assignment
    (Hungarian / linear_sum_assignment).
  * Life cycle: tentative -> confirmed after N hits; deleted after M misses.
  * Fusion: each sensor's detections are applied as a sequential update with
    that sensor's own noise. LiDAR also contributes an orientation measurement
    (L-shape yaw, 180-degree ambiguity resolved against the track). The camera
    contributes semantics: canonical class and open-vocabulary label.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from avstack.common.geometry import normalize_angle
from avstack.common.types import Detection, ObjectClass, Track
from avstack.config import TrackingConfig

_H_YAW = np.array([[0, 0, 0, 1.0, 0]])


def ctrv_step(x: np.ndarray, dt: float) -> np.ndarray:
    px, py, v, yaw, w = x
    if abs(w) > 1e-4:
        px += v / w * (np.sin(yaw + w * dt) - np.sin(yaw))
        py += v / w * (-np.cos(yaw + w * dt) + np.cos(yaw))
    else:
        px += v * np.cos(yaw) * dt
        py += v * np.sin(yaw) * dt
    return np.array([px, py, v, normalize_angle(yaw + w * dt), w])


def _ctrv_jacobian(x: np.ndarray, dt: float, eps: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian: robust through the w -> 0 singularity."""
    f0 = ctrv_step(x, dt)
    J = np.zeros((5, 5))
    for i in range(5):
        dx = np.zeros(5)
        dx[i] = eps
        diff = ctrv_step(x + dx, dt) - f0
        diff[3] = normalize_angle(diff[3])
        J[:, i] = diff / eps
    return J


def motion_model_for(cls: ObjectClass, length: float) -> str:
    """Vehicles steer (non-holonomic: CTRV). Pedestrians, animals, cones and small
    unknown blobs can move in any direction (holonomic: constant velocity)."""
    if cls.family == "vehicle" or (cls.family == "unknown" and length >= 3.0):
        return "ctrv"
    return "cv"


class _Track:
    """One tracked object with a class-dependent motion model.

    CTRV state: [x, y, v, yaw, yaw_rate]      CV state: [x, y, vx, vy]
    Both expose the same interface; the model switches if the class estimate
    changes (e.g. a LiDAR blob later labelled "pedestrian" by the camera).
    """

    def __init__(self, tid: int, det: Detection, cfg: TrackingConfig, v0: float, yaw0: float):
        self.id = tid
        self.length, self.width, self.cls = det.length, det.width, det.cls
        self.label = det.label
        self.cls_votes: dict[ObjectClass, float] = {} if det.cls == ObjectClass.UNKNOWN else {det.cls: det.score}
        self.hits, self.misses, self.age = 1, 0, 0
        self.confirmed = False
        self.sources: set[str] = {det.source}
        self.model = motion_model_for(det.cls, det.length)
        if self.model == "ctrv":
            self.x = np.array([det.x, det.y, v0, yaw0, 0.0])
            v_std = cfg.init_vel_std if det.cls.family != "static" else 1.0
            self.P = np.diag([det.pos_std**2, det.pos_std**2, v_std**2, 0.3**2, 0.2**2])
        else:
            self.x = np.array([det.x, det.y, v0 * np.cos(yaw0), v0 * np.sin(yaw0)])
            v_std = 0.5 if det.cls.family == "static" else 3.0
            self.P = np.diag([det.pos_std**2, det.pos_std**2, v_std**2, v_std**2])

    # -- kinematics independent of the model ------------------------------ #
    @property
    def velocity(self) -> np.ndarray:
        if self.model == "ctrv":
            return self.x[2] * np.array([np.cos(self.x[3]), np.sin(self.x[3])])
        return self.x[2:4].copy()

    @property
    def yaw(self) -> float:
        if self.model == "ctrv":
            return float(self.x[3])
        vx, vy = self.x[2:4]
        return float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > 0.3 else getattr(self, "_last_yaw", 0.0)

    @property
    def speed_std(self) -> float:
        if self.model == "ctrv":
            return float(np.sqrt(max(self.P[2, 2], 0.0)))
        return float(np.sqrt(max(0.5 * (self.P[2, 2] + self.P[3, 3]), 0.0)))

    @property
    def yaw_rate(self) -> float:
        return float(self.x[4]) if self.model == "ctrv" else 0.0

    def _switch_model(self, model: str) -> None:
        if model == self.model:
            return
        vel, pos_cov = self.velocity, self.P[:2, :2].copy()
        if model == "cv":
            self.x = np.array([self.x[0], self.x[1], vel[0], vel[1]])
            self.P = np.zeros((4, 4))
            self.P[2:, 2:] = np.eye(2) * 3.0**2
        else:
            spd = float(np.hypot(*vel))
            self.x = np.array([self.x[0], self.x[1], spd, np.arctan2(vel[1], vel[0]) if spd > 0.3 else 0.0, 0.0])
            self.P = np.zeros((5, 5))
            self.P[2:, 2:] = np.diag([4.0**2, 0.5**2, 0.2**2])
        self.P[:2, :2] = pos_cov
        self.model = model

    # -- filter ---------------------------------------------------------- #
    def predict(self, dt: float, process_noise: dict) -> None:
        sa, sw = process_noise[self.cls.family]
        if self.model == "ctrv":
            F = _ctrv_jacobian(self.x, dt)
            self.x = ctrv_step(self.x, dt)
            c, s = np.cos(self.x[3]), np.sin(self.x[3])
            G = np.array([[0.5 * dt**2 * c, 0], [0.5 * dt**2 * s, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
            self.P = F @ self.P @ F.T + G @ np.diag([sa**2, sw**2]) @ G.T
            if self.cls.family == "static":
                self.x[2] *= 0.9
        else:
            self._last_yaw = self.yaw
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + G @ G.T * sa**2
            if self.cls.family == "static":
                self.x[2:] *= 0.8                  # cones and debris do not move

    def _kf_update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        n = len(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        if self.model == "ctrv":
            self.x[3] = normalize_angle(self.x[3])
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def _h_pos(self) -> np.ndarray:
        H = np.zeros((2, len(self.x)))
        H[0, 0] = H[1, 1] = 1.0
        return H

    def recenter(self, det: Detection, sensor_xy: np.ndarray | None) -> Detection:
        """A LiDAR box shorter than the track is a partial view (e.g. only a truck's
        tail is visible). Its visible end is reliable; slide the centre away from
        the sensor so the box has the track's known length."""
        length = self.length
        if sensor_xy is None or det.source != "lidar" or det.length > length - 1.0:
            return det
        heading = np.array([np.cos(det.yaw), np.sin(det.yaw)])     # L-shape yaw beats velocity yaw
        away = np.sign((np.array([det.x, det.y]) - sensor_xy) @ heading) or 1.0
        shift = away * (length - det.length) / 2 * heading
        return Detection(det.x + shift[0], det.y + shift[1], det.yaw, length, det.width,
                         det.cls, det.source, det.pos_std, det.label, det.score)

    def innovation(self, det: Detection):
        H = self._h_pos()
        y = np.array([det.x, det.y]) - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * det.pos_std**2
        return y, S

    def update_range_rate(self, det: Detection, gate: float = 25.0) -> None:
        """Doppler update: h(x) = (v_obj - v_sensor) . u, u = line of sight."""
        los = np.array([self.x[0] - det.sensor_xy[0], self.x[1] - det.sensor_xy[1]])
        los /= max(np.linalg.norm(los), 1e-6)
        pred = float((self.velocity - np.array(det.sensor_vel)) @ los)
        if self.model == "ctrv":
            v, yaw = self.x[2], self.x[3]
            heading = np.array([np.cos(yaw), np.sin(yaw)])
            H = np.array([[0.0, 0.0, heading @ los, v * (np.array([-np.sin(yaw), np.cos(yaw)]) @ los), 0.0]])
        else:
            H = np.array([[0.0, 0.0, los[0], los[1]]])
        R = np.array([[0.15**2]])
        y = np.array([det.range_rate - pred])
        S = float((H @ self.P @ H.T + R)[0, 0])
        if y[0] ** 2 / S < gate:
            self._kf_update(y, H, R)

    def update(self, det: Detection) -> None:
        y, _ = self.innovation(det)
        self._kf_update(y, self._h_pos(), np.eye(2) * det.pos_std**2)
        if det.range_rate is not None:
            self.update_range_rate(det)
        if det.source == "lidar" and self.model == "ctrv" and self.x[2] > -1.0:
            # L-shape orientation, resolved for the 180-degree box ambiguity
            dyaw = normalize_angle(det.yaw - self.x[3])
            if abs(dyaw) > np.pi / 2:
                dyaw = normalize_angle(dyaw - np.pi)
            if abs(dyaw) < 0.5:
                self._kf_update(np.array([dyaw]), _H_YAW, np.array([[0.08**2]]))
        # shape: only LiDAR measures extent (radar has none, mono-camera boxes are class priors)
        alpha = {"lidar": 0.3, "camera": 0.05}.get(det.source, 0.0)
        self.length += alpha * (det.length - self.length)
        self.width += alpha * (det.width - self.width)
        # semantics: accumulate class evidence, keep the most confident open-vocab label
        if det.label and det.cls != ObjectClass.UNKNOWN:            # camera semantics present
            self.cls_votes[det.cls] = self.cls_votes.get(det.cls, 0.0) + det.score
            self.cls = max(self.cls_votes, key=self.cls_votes.get)
            self.label = det.label
        elif self.cls == ObjectClass.UNKNOWN and det.cls != ObjectClass.UNKNOWN:
            self.cls = det.cls
        max_l, max_w = self.cls.max_dims
        self.length, self.width = min(self.length, max_l), min(self.width, max_w)
        self._switch_model(motion_model_for(self.cls, self.length))
        self.sources.add(det.source)
        self.hits += 1


class MultiObjectTracker:
    def __init__(self, cfg: TrackingConfig, velocity_prior=None, lane_heading=None):
        """``velocity_prior(x, y, cls) -> speed`` seeds new tracks (vehicles move with
        traffic, cones do not) so they do not start 'parked' and confuse prediction.
        ``lane_heading(x, y) -> yaw | None`` seeds the heading from the HD map."""
        self.cfg = cfg
        self.velocity_prior = velocity_prior
        self.lane_heading = lane_heading
        self.tracks: list[_Track] = []
        self.t: float | None = None
        self._ids = itertools.count(1)

    def predict(self, t: float) -> None:
        if self.t is not None and t > self.t:
            for tr in self.tracks:
                tr.predict(t - self.t, self.cfg.process_noise)
        self.t = t

    def update(self, t: float, detections: list[Detection], sensor: str,
               sensor_xy: np.ndarray | None = None) -> None:
        self.predict(t)
        cfg = self.cfg
        n, m = len(self.tracks), len(detections)
        matched_tracks, matched_dets = set(), set()
        if n and m:
            cost = np.full((n, m), 1e6)
            adjusted = [[tr.recenter(det, sensor_xy) for det in detections] for tr in self.tracks]
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(adjusted[i]):
                    y, S = tr.innovation(det)
                    d2 = float(y @ np.linalg.solve(S, y))
                    max_dist = max(cfg.max_assoc_dist, 3.0 * det.pos_std)
                    if d2 < cfg.gate_chi2 and float(np.hypot(*y)) < max_dist:
                        cost[i, j] = d2 + np.log(np.linalg.det(S))   # prefer confident matches
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e5:
                    self.tracks[i].update(adjusted[i][j])
                    matched_tracks.add(i)
                    matched_dets.add(j)

        for i, tr in enumerate(self.tracks):
            tr.age += 1
            if i in matched_tracks:
                tr.misses = 0
                if tr.hits >= cfg.confirm_hits:
                    tr.confirmed = True
            elif sensor == "lidar":
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks
                       if tr.misses < (cfg.max_misses if tr.confirmed else 2)]
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            if det.range_rate is not None and self._ground_speed(det) < 2.0:
                continue      # stationary radar-only returns: manholes and guard rails, not new objects
            self.tracks.append(self._new_track(det))
        self._merge_duplicates()

    @staticmethod
    def _radial_ground_speed(det: Detection) -> tuple[float, np.ndarray]:
        los = np.array([det.x - det.sensor_xy[0], det.y - det.sensor_xy[1]])
        los /= max(np.linalg.norm(los), 1e-6)
        return det.range_rate + float(np.array(det.sensor_vel) @ los), los

    def _ground_speed(self, det: Detection) -> float:
        return abs(self._radial_ground_speed(det)[0])

    def _new_track(self, det: Detection) -> _Track:
        yaw0 = self.lane_heading(det.x, det.y) if self.lane_heading else None
        if yaw0 is None:
            yaw0 = det.yaw
        v0 = self.velocity_prior(det.x, det.y, det.cls, det.length) if self.velocity_prior else 0.0
        if det.range_rate is not None:
            # Doppler gives the radial ground speed; project onto the lane direction
            radial, los = self._radial_ground_speed(det)
            along = float(np.array([np.cos(yaw0), np.sin(yaw0)]) @ los)
            if abs(along) > 0.5:
                v0 = radial / along
        tr = _Track(next(self._ids), det, self.cfg, v0, yaw0)
        if det.source == "camera" and det.pos_std > 2.0:
            tr.P[2, 2] = 15.0**2        # far mono-only: velocity genuinely unknown
        return tr

    def _merge_duplicates(self) -> None:
        """Two tracks on one object (split clusters, camera/LiDAR offset): keep the elder."""
        drop = set()
        for a, b in itertools.combinations(range(len(self.tracks)), 2):
            ta, tb = self.tracks[a], self.tracks[b]
            if a in drop or b in drop:
                continue
            gap = np.hypot(*(ta.x[:2] - tb.x[:2]))
            overlap = gap < 0.5 * (max(ta.length, tb.length) + min(ta.width, tb.width)) * 0.6
            if ta.cls.family == "static" and tb.cls.family == "static":
                # far camera-only statics: positions are uncertain by metres, so "same
                # object" means "closer than the combined 2-sigma position uncertainty"
                sig = np.sqrt(max(ta.P[0, 0], ta.P[1, 1]) + max(tb.P[0, 0], tb.P[1, 1]))
                overlap = overlap or gap < max(1.5, 2.0 * sig)
            if overlap and np.hypot(*(ta.velocity - tb.velocity)) < 4.0:
                drop.add(b if ta.hits >= tb.hits else a)
        if drop:
            self.tracks = [t for i, t in enumerate(self.tracks) if i not in drop]

    def confirmed_tracks(self) -> list[Track]:
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            vx, vy = tr.velocity
            out.append(Track(id=tr.id, x=float(tr.x[0]), y=float(tr.x[1]), vx=float(vx), vy=float(vy),
                             yaw=tr.yaw, length=tr.length, width=tr.width, cls=tr.cls, cov=tr.P.copy(),
                             confirmed=True, age=tr.age, yaw_rate=tr.yaw_rate, label=tr.label,
                             sources=tuple(sorted(tr.sources)), hits=tr.hits, speed_std=tr.speed_std))
        return out
