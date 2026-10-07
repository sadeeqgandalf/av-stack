"""Ego plant model: kinematic bicycle + actuator dynamics.

The controller never sees this model directly; it only gets noisy sensor data.
Actuator lag and steering-rate limits make the control problem realistic.
"""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import normalize_angle
from avstack.common.types import ControlCommand, VehicleState
from avstack.config import VehicleConfig


class KinematicBicycle:
    def __init__(self, cfg: VehicleConfig, state: VehicleState):
        self.cfg = cfg
        self.state = state

    def step(self, cmd: ControlCommand, dt: float) -> VehicleState:
        c, s = self.cfg, self.state
        # longitudinal actuator: first-order lag toward the commanded accel
        a_cmd = float(np.clip(cmd.accel, -c.max_decel, c.max_accel))
        s.a += (a_cmd - s.a) * min(dt / c.actuator_tau, 1.0)
        # steering actuator: first-order lag toward the command, then the motor's rate limit
        # (applying the rate limit *before* the lag would compound them into a far slower actuator)
        steer_cmd = float(np.clip(cmd.steer, -c.max_steer, c.max_steer))
        desired_delta = (steer_cmd - s.steer) * min(dt / c.actuator_tau, 1.0)
        max_delta = c.max_steer_rate * dt
        s.steer += float(np.clip(desired_delta, -max_delta, max_delta))

        v_new = max(0.0, s.v + s.a * dt)
        if v_new == 0.0 and s.a < 0:          # a stopped car does not reverse
            s.a = 0.0
        v_mid = 0.5 * (s.v + v_new)
        s.yaw_rate = v_mid / c.wheelbase * np.tan(s.steer)
        yaw_mid = s.yaw + 0.5 * s.yaw_rate * dt
        s.x += v_mid * np.cos(yaw_mid) * dt
        s.y += v_mid * np.sin(yaw_mid) * dt
        s.yaw = float(normalize_angle(s.yaw + s.yaw_rate * dt))
        s.v = v_new
        return s
