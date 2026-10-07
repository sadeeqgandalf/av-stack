"""Trajectory tracking control.

Lateral (selectable in ControlConfig):
  * LQR (default): speed-scheduled, actuator-lag-aware, curvature feed-forward.
  * Stanley: front-axle cross-track + heading error + curvature feed-forward.
  * Pure Pursuit: geometric lookahead.
Longitudinal: PID on speed error + trajectory acceleration feed-forward,
          previewed by the actuator lag so the car does not respond late.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from avstack.common.geometry import normalize_angle
from avstack.common.types import ControlCommand, Trajectory, VehicleState
from avstack.config import ControlConfig, VehicleConfig


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float = 2.0):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.integral = 0.0
        self.prev_err: float | None = None

    def step(self, err: float, dt: float) -> float:
        self.integral = float(np.clip(self.integral + err * dt, -self.i_limit, self.i_limit))
        deriv = 0.0 if self.prev_err is None or dt <= 0 else (err - self.prev_err) / dt
        self.prev_err = err
        return self.kp * err + self.ki * self.integral + self.kd * deriv


class LateralLQR:
    """Discrete LQR on the path-error dynamics of a kinematic bicycle (rear axle).

        state  z = [e, psi_e, delta]   lateral error (+ left), heading error
                                       (vehicle - path), actual wheel angle
        input  u = delta_cmd           commanded wheel angle (feedback part)

        e_dot     = v * psi_e
        psi_e_dot = v / L * delta - v * kappa        (kappa handled by feed-forward)
        delta_dot = (delta_cmd - delta) / tau        (steering actuator lag)

    Gains depend on speed, so the Riccati equation is solved per 1 m/s bin and
    cached: stiff at parking speeds, gentle and well-damped at highway speeds.
    """

    def __init__(self, wheelbase: float, tau: float, dt: float = 0.02,
                 q=(1.0, 1.0, 0.0), r: float = 0.1):
        self.L, self.tau, self.dt = wheelbase, tau, dt
        self.Q, self.R = np.diag(q), np.array([[r]])
        self._cache: dict[int, np.ndarray] = {}

    def gain(self, v: float) -> np.ndarray:
        key = int(np.clip(round(v), 1, 45))
        if key not in self._cache:
            vv, L, tau, dt = max(float(key), 1.0), self.L, self.tau, self.dt
            A = np.array([[0.0, vv, 0.0], [0.0, 0.0, vv / L], [0.0, 0.0, -1.0 / tau]])
            B = np.array([[0.0], [0.0], [1.0 / tau]])
            Ad, Bd = np.eye(3) + A * dt, B * dt
            # Penalise the *lateral acceleration* the input produces (a_y = v^2/L * delta),
            # not the raw wheel angle: comfort is then enforced consistently at every speed.
            # below ~4 m/s the kinematics give little lateral authority, so an unweighted LQR
            # demands huge wheel angles and amplifies localisation noise: floor the weighting
            R = self.R * max(vv, 4.0) ** 4 / L**2
            P = solve_discrete_are(Ad, Bd, self.Q, R)
            self._cache[key] = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return self._cache[key]

    def steer(self, e: float, psi_e: float, delta: float, v: float, kappa: float) -> float:
        K = self.gain(v)
        ff = np.arctan(self.L * kappa)
        return float(ff - (K @ np.array([e, psi_e, delta - ff]))[0])


class TrajectoryController:
    def __init__(self, cfg: ControlConfig, vcfg: VehicleConfig):
        self.cfg, self.vcfg = cfg, vcfg
        self.speed_pid = PID(cfg.kp, cfg.ki, cfg.kd)
        self.lqr = LateralLQR(vcfg.wheelbase, vcfg.actuator_tau)
        self.last_steer = 0.0
        self.last_t: float | None = None
        self.last_errors = {"cross_track": 0.0, "heading": 0.0, "speed": 0.0}

    @staticmethod
    def _nearest(traj: Trajectory, px: float, py: float) -> tuple[int, float, float]:
        """Closest path segment, the projection fraction on it, and the signed lateral error.

        Stopping plans bunch their samples at the stop point (zero-length segments), so
        coincident samples are dropped, and the error is the perpendicular distance to the
        segment's *line*: past either end of the path, the distance to the end point would be
        mostly longitudinal and read as a large, arbitrary-signed cross-track error.
        """
        pts = np.stack([traj.x, traj.y], axis=1)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.05])
        idx = np.flatnonzero(keep)
        p = np.array([px, py])
        if len(idx) < 2:                       # the whole plan is (nearly) one point
            yaw = traj.yaw[0]
            d = p - pts[0]
            return 0, 0.0, float(-d[0] * np.sin(yaw) + d[1] * np.cos(yaw))
        q = pts[idx]
        a, b = q[:-1], q[1:]
        ab = b - a
        L2 = np.einsum("ij,ij->i", ab, ab)
        u_raw = np.einsum("ij,ij->i", p - a, ab) / L2
        u = np.clip(u_raw, 0, 1)
        dist = np.linalg.norm(p - (a + u[:, None] * ab), axis=1)
        k = int(np.argmin(dist))
        cross = ab[k, 0] * (py - a[k, 1]) - ab[k, 1] * (px - a[k, 0])
        e = float(cross / np.sqrt(L2[k]))      # signed perpendicular distance to the line
        i = int(idx[k])
        j = int(idx[k + 1])
        frac = float(np.clip(u_raw[k], 0, 1))
        # map the fraction back onto the original (i, i+1) indexing used by callers
        if j - i > 1:
            pos = i + frac * (j - i)
            i, frac = int(min(np.floor(pos), len(traj.x) - 2)), float(pos - np.floor(pos))
        return min(i, len(traj.x) - 2), frac, e

    def _stanley(self, traj: Trajectory, ego: VehicleState) -> float:
        c, L = self.cfg, self.vcfg.wheelbase
        # rear-axle state -> front axle (ego state is the vehicle reference point = rear axle)
        fx, fy = ego.x + L * np.cos(ego.yaw), ego.y + L * np.sin(ego.yaw)
        i, u, e = self._nearest(traj, fx, fy)
        yaw_ref = traj.yaw[i] + u * normalize_angle(traj.yaw[i + 1] - traj.yaw[i])
        kappa = traj.kappa[i] + u * (traj.kappa[i + 1] - traj.kappa[i])
        heading_err = float(normalize_angle(yaw_ref - ego.yaw))
        self.last_errors.update(cross_track=e, heading=heading_err)
        return (c.heading_gain * heading_err
                + np.arctan2(-c.stanley_k * e, c.stanley_soft + ego.v)
                + np.arctan(L * kappa))

    def _lqr(self, traj: Trajectory, ego: VehicleState) -> float:
        i, u, e = self._nearest(traj, ego.x, ego.y)
        yaw_ref = traj.yaw[i] + u * normalize_angle(traj.yaw[i + 1] - traj.yaw[i])
        kappa = traj.kappa[i] + u * (traj.kappa[i + 1] - traj.kappa[i])
        psi_e = float(normalize_angle(ego.yaw - yaw_ref))
        self.last_errors.update(cross_track=e, heading=-psi_e)
        # the actual wheel angle is not measured here: use our last command filtered by the lag
        return self.lqr.steer(e, psi_e, self.last_steer, ego.v, kappa)

    def _pure_pursuit(self, traj: Trajectory, ego: VehicleState) -> float:
        c, L = self.cfg, self.vcfg.wheelbase
        i, _, e = self._nearest(traj, ego.x, ego.y)
        self.last_errors.update(cross_track=e, heading=float(normalize_angle(traj.yaw[i] - ego.yaw)))
        Ld = max(c.pp_min_lookahead, c.pp_lookahead_gain * ego.v)
        d = np.hypot(traj.x[i:] - ego.x, traj.y[i:] - ego.y)
        j = i + int(np.argmax(d >= Ld)) if np.any(d >= Ld) else len(traj) - 1
        alpha = normalize_angle(np.arctan2(traj.y[j] - ego.y, traj.x[j] - ego.x) - ego.yaw)
        return float(np.arctan2(2.0 * L * np.sin(alpha), max(d[j - i], 1e-3)))

    def compute(self, t: float, ego: VehicleState, traj: Trajectory) -> ControlCommand:
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        lateral = {"lqr": self._lqr, "stanley": self._stanley, "pure_pursuit": self._pure_pursuit}
        steer = lateral[self.cfg.lateral](traj, ego)
        steer = float(np.clip(steer, -self.vcfg.max_steer, self.vcfg.max_steer))
        # track the actuator's state for the LQR's wheel-angle estimate: first-order lag *and*
        # the steering-rate limit, or the estimate diverges from the real wheel at low speed
        if dt > 0:
            want = (steer - self.last_steer) * min(dt / self.vcfg.actuator_tau, 1.0)
            lim = self.vcfg.max_steer_rate * dt
            self.last_steer += float(np.clip(want, -lim, lim))

        ref = traj.sample(t + self.vcfg.actuator_tau)
        err = ref["v"] - ego.v
        self.last_errors["speed"] = err
        accel = ref["a"] + self.speed_pid.step(err, dt)
        if ref["v"] < 0.1 and ego.v < 0.3:
            accel = -1.0                                  # hold the brake at standstill
            self.speed_pid.integral = 0.0
        return ControlCommand(accel=float(np.clip(accel, -self.vcfg.max_decel, self.vcfg.max_accel)),
                              steer=steer, source="planner")
