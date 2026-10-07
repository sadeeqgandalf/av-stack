"""Rate scheduling helpers for a multi-rate pipeline inside a fixed-step loop."""
from __future__ import annotations


class RateGate:
    """Fires at a fixed rate inside a faster simulation loop."""

    def __init__(self, rate: float, phase: float = 0.0):
        self.period = 1.0 / rate
        self.next_t = phase

    def ready(self, t: float) -> bool:
        if t + 1e-9 >= self.next_t:
            self.next_t += self.period
            return True
        return False
