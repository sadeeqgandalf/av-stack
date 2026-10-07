"""Jerk-optimal polynomials for Frenet trajectory generation (Werling et al. 2010)."""
from __future__ import annotations

import numpy as np


class QuinticPolynomial:
    """x(t) with boundary conditions on position, velocity, acceleration at 0 and T."""

    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.c = np.zeros(6)
        self.c[:3] = [x0, v0, a0 / 2.0]
        A = np.array([[T**3, T**4, T**5],
                      [3 * T**2, 4 * T**3, 5 * T**4],
                      [6 * T, 12 * T**2, 20 * T**3]])
        b = np.array([x1 - x0 - v0 * T - a0 / 2 * T**2,
                      v1 - v0 - a0 * T,
                      a1 - a0])
        self.c[3:] = np.linalg.solve(A, b)

    def eval(self, t, order: int = 0):
        return np.polyval(np.polyder(self.c[::-1], order) if order else self.c[::-1], t)


class QuarticPolynomial:
    """Velocity keeping: fixes x0, v0, a0 and v(T), a(T); final position is free."""

    def __init__(self, x0, v0, a0, v1, a1, T):
        self.c = np.zeros(5)
        self.c[:3] = [x0, v0, a0 / 2.0]
        A = np.array([[3 * T**2, 4 * T**3], [6 * T, 12 * T**2]])
        b = np.array([v1 - v0 - a0 * T, a1 - a0])
        self.c[3:] = np.linalg.solve(A, b)

    def eval(self, t, order: int = 0):
        return np.polyval(np.polyder(self.c[::-1], order) if order else self.c[::-1], t)


class ImmediateBrake:
    """Emergency primitive: ramp to ``decel`` within ``ramp`` seconds, then hold it until stopped.

    Jerk-optimal polynomials start at the current acceleration (~0) and ease into braking,
    which is right for comfort and wrong when a child steps out. This one bites at once.
    """

    def __init__(self, x0, v0, decel, ramp: float = 0.15, dt: float = 0.01):
        t = np.arange(0.0, 30.0, dt)
        a = -decel * np.minimum(t / ramp, 1.0)
        v = np.maximum(v0 + np.cumsum(a) * dt, 0.0)
        a = np.where(v > 0, a, 0.0)
        x = x0 + np.cumsum(v) * dt
        j = np.gradient(a, dt)
        self._t, self._cols = t, (x, v, a, j)

    def eval(self, t, order: int = 0):
        return np.interp(t, self._t, self._cols[order])


def eval_with_hold(poly, ts: np.ndarray, T: float):
    """Evaluate position/vel/acc/jerk on ``ts``; beyond T hold the final velocity."""
    inside = ts <= T
    x, v, a, j = (np.empty_like(ts) for _ in range(4))
    x[inside], v[inside] = poly.eval(ts[inside]), poly.eval(ts[inside], 1)
    a[inside], j[inside] = poly.eval(ts[inside], 2), poly.eval(ts[inside], 3)
    xT, vT = poly.eval(T), poly.eval(T, 1)
    dt = ts[~inside] - T
    x[~inside], v[~inside], a[~inside], j[~inside] = xT + vT * dt, vT, 0.0, 0.0
    return x, v, a, j
