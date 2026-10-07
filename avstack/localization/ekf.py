"""Ego localization: Extended Kalman Filter fusing IMU, wheel odometry and GNSS.

State  x = [px, py, yaw, v, b_g]   (b_g = gyro yaw-rate bias, estimated online)
Predict with IMU (yaw rate, longitudinal accel) at the sim rate.
Update  with wheel speed (every tick) and GNSS position (10 Hz, chi^2 gated).
"""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import normalize_angle
from avstack.common.types import GnssMsg, ImuMsg, LaneObservation, VehicleState, WheelOdomMsg
from avstack.config import LocalizationConfig, SensorConfig


class LocalizationEKF:
    def __init__(self, cfg: LocalizationConfig, sensor_cfg: SensorConfig, init: VehicleState,
                 hd_map=None):
        self.cfg = cfg
        self.map = hd_map
        self.lane_accepted = 0
        self.scfg = sensor_cfg
        self.x = np.array([init.x, init.y, init.yaw, init.v, 0.0])
        self.P = np.diag([cfg.init_pos_std**2, cfg.init_pos_std**2, cfg.init_yaw_std**2, 1.0, 1e-4])
        self.last_yaw_rate = 0.0
        self.last_accel = 0.0
        self.gnss_rejected = 0
        self.gnss_accepted = 0
        self._consecutive_rejects = 0

    # ------------------------------------------------------------------ #
    def predict(self, imu: ImuMsg, dt: float) -> None:
        px, py, yaw, v, b = self.x
        w = imu.yaw_rate - b
        self.last_yaw_rate, self.last_accel = w, imu.accel
        self.x = np.array([px + v * np.cos(yaw) * dt,
                           py + v * np.sin(yaw) * dt,
                           normalize_angle(yaw + w * dt),
                           max(0.0, v + imu.accel * dt),
                           b])
        F = np.eye(5)
        F[0, 2], F[0, 3] = -v * np.sin(yaw) * dt, np.cos(yaw) * dt
        F[1, 2], F[1, 3] = v * np.cos(yaw) * dt, np.sin(yaw) * dt
        F[2, 4] = -dt
        c, s = self.cfg, self.scfg
        Q = np.diag([c.q_pos * dt, c.q_pos * dt,
                     (s.imu_yaw_rate_std**2 + c.q_yaw) * dt,
                     (s.imu_accel_std**2 + c.q_v) * dt,
                     c.q_bias * dt])
        self.P = F @ self.P @ F.T + Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, gate: float | None = None) -> bool:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        if gate is not None and float(y @ Sinv @ y) > gate:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.x[2] = normalize_angle(self.x[2])
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form: stays symmetric PSD
        return True

    def update_odom(self, odom: WheelOdomMsg) -> None:
        H = np.array([[0, 0, 0, 1.0, 0]])
        self._update(np.array([odom.v]), H, np.array([[self.scfg.odom_std**2]]))

    def update_gnss(self, gnss: GnssMsg) -> bool:
        H = np.zeros((2, 5))
        H[0, 0] = H[1, 1] = 1.0
        z, R = np.array([gnss.x, gnss.y]), np.eye(2) * gnss.std**2
        ok = self._update(z, H, R, gate=self.cfg.gnss_gate_chi2)
        if ok:
            self.gnss_accepted += 1
            self._consecutive_rejects = 0
            return True
        self.gnss_rejected += 1
        self._consecutive_rejects = self._consecutive_rejects + 1
        if self._consecutive_rejects >= 5:
            # Persistent disagreement means *we* drifted, not the GNSS: re-open the filter.
            self.P[:2, :2] += np.eye(2) * 4.0
            self._update(z, H, R)
            self._consecutive_rejects = 0
        return False

    def update_lane(self, obs: LaneObservation) -> bool:
        """Map matching: compare observed lane-relative pose with the pose predicted
        from our estimate + HD map. Gives cm-level lateral and sub-degree heading.

        h(x) = [ d(px, py) - centre(lane),  yaw - lane_yaw(s) ]
        dd/dpx = -sin(lane_yaw), dd/dpy = cos(lane_yaw)
        Lane ambiguity: the camera cannot tell *which* lane we are in, so the
        offset innovation is wrapped to +-lane_width/2 (GNSS resolves the lane).
        """
        if self.map is None:
            return False
        ref, m = self.map.reference, self.map
        s, d = ref.project(self.x[0], self.x[1])
        lane_yaw = float(ref.interpolate(s)[2])
        pred = np.array([d - m.lane_center_d(m.lane_at(d)), normalize_angle(self.x[2] - lane_yaw)])
        y = np.array([obs.offset, obs.heading]) - pred
        y[0] = (y[0] + m.lane_width / 2) % m.lane_width - m.lane_width / 2
        y[1] = normalize_angle(y[1])
        H = np.zeros((2, 5))
        H[0, 0], H[0, 1] = -np.sin(lane_yaw), np.cos(lane_yaw)
        H[1, 2] = 1.0
        R = np.diag([obs.offset_std**2, obs.heading_std**2])
        # _update() expects z; express the wrapped innovation as z = Hx + y
        ok = self._update(H @ self.x + y, H, R, gate=self.cfg.lane_gate_chi2)
        self.lane_accepted += ok
        return ok

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> VehicleState:
        px, py, yaw, v, _ = self.x
        return VehicleState(x=float(px), y=float(py), yaw=float(yaw), v=float(v),
                            a=self.last_accel, yaw_rate=self.last_yaw_rate)
