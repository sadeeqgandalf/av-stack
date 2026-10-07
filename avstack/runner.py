"""Closed-loop simulation runner + evaluation metrics."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from avstack.common.geometry import box_corners, polygon_distance
from avstack.common.types import ControlCommand, VehicleState
from avstack.config import StackConfig
from avstack.pipeline import AVStack
from avstack.sim.scenarios import Scenario
from avstack.sim.sensors import SensorSuite
from avstack.sim.world import World


@dataclass
class RunResult:
    scenario: str
    success: bool
    collided: bool
    off_road: bool
    collision_speed: float
    collision_with: str
    reached_goal: bool
    stalled: bool
    sim_time: float
    distance: float
    loc_rmse: float
    loc_max_err: float
    loc_lat_rmse: float
    loc_lon_rmse: float
    tracking_rmse: float
    mean_cross_track: float
    max_cross_track: float
    max_lat_accel: float
    max_jerk: float
    min_ttc: float
    min_clearance: dict
    aeb_events: int
    gnss_rejected: int
    timing_ms: dict
    history: dict = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.success else "FAIL"
        lines = [
            f"[{status}] {self.scenario}: {self.sim_time:.1f}s, {self.distance:.0f} m driven",
            f"  safety      collided={self.collided}"
            + (f" ({self.collision_with!r} at {self.collision_speed:.1f} m/s)" if self.collided else "")
            + f" off_road={self.off_road} "
            f"reached_goal={self.reached_goal} stalled={self.stalled} min_TTC={self.min_ttc:.2f}s "
            f"AEB_events={self.aeb_events}",
            "  clearance   " + (", ".join(f"{k}={v:.2f}m" for k, v in sorted(self.min_clearance.items()))
                                or "n/a"),
            f"  localization RMSE={self.loc_rmse:.2f} m (lateral {self.loc_lat_rmse:.2f}, "
            f"longitudinal {self.loc_lon_rmse:.2f}, max {self.loc_max_err:.2f} m), "
            f"GNSS outliers rejected={self.gnss_rejected}",
            f"  tracking    position RMSE={self.tracking_rmse:.2f} m",
            f"  control     cross-track mean={self.mean_cross_track:.2f} m max={self.max_cross_track:.2f} m",
            f"  comfort     max |lat accel|={self.max_lat_accel:.2f} m/s^2, max |jerk|={self.max_jerk:.1f} m/s^3",
            "  compute     " + ", ".join(f"{k}={m:.1f}ms" for k, (m, _) in self.timing_ms.items()),
        ]
        return "\n".join(lines)


def build(scenario: Scenario, cfg: StackConfig | None = None, seed: int = 0):
    cfg = cfg or StackConfig()
    rng = np.random.default_rng(seed)
    true_state = scenario.ego_state(cfg.vehicle.rear_to_center)
    world = World(scenario.map, true_state, copy.deepcopy(scenario.actors), cfg.vehicle)
    sensors = SensorSuite(cfg.sensors, rng)
    # the stack starts from a noisy prior, as if from a cold GNSS fix
    guess = VehicleState(x=true_state.x + rng.normal(0, 0.8), y=true_state.y + rng.normal(0, 0.8),
                         yaw=true_state.yaw + rng.normal(0, 0.03), v=true_state.v)
    stack = AVStack(cfg, scenario.map, guess, scenario.goal_s, scenario.goal_lane)
    return world, sensors, stack, cfg


def run(scenario: Scenario, cfg: StackConfig | None = None, seed: int = 0, duration: float | None = None,
        on_step=None, stop_on_collision: bool = True) -> RunResult:
    world, sensors, stack, cfg = build(scenario, cfg, seed)
    dt = cfg.sim_dt
    duration = duration or scenario.duration
    H = {k: [] for k in ("t", "x", "y", "v", "a", "yaw_rate", "steer", "est_x", "est_y",
                         "yaw", "cross_track", "cmd_accel", "behavior", "ttc", "track_err")}
    start_s, _ = world.ego_frenet()
    cmd = ControlCommand()
    min_clear: dict[str, float] = {}
    reached = False
    for _ in range(int(duration / dt)):
        bundle = sensors.measure(world)
        cmd = stack.step(bundle)
        world.step(cmd, dt)

        e, est, dbg = world.ego.state, stack.debug.ego_estimate, stack.debug
        H["t"].append(world.t); H["x"].append(e.x); H["y"].append(e.y); H["v"].append(e.v)
        H["a"].append(e.a); H["yaw_rate"].append(e.yaw_rate); H["steer"].append(e.steer)
        H["est_x"].append(est.x); H["est_y"].append(est.y); H["yaw"].append(e.yaw)
        # tracking error is only meaningful while moving (a stopped car has no path to follow)
        H["cross_track"].append(stack.controller.last_errors["cross_track"] if e.v > 1.0 else 0.0)
        H["cmd_accel"].append(cmd.accel); H["behavior"].append(dbg.behavior); H["ttc"].append(dbg.min_ttc)
        for tr in (dbg.tracks if world.actors else []):   # track error vs nearest ground truth
            gt = min(world.actors, key=lambda a: np.hypot(a.x - tr.x, a.y - tr.y))
            err = np.hypot(gt.x - tr.x, gt.y - tr.y)
            if err < 5.0:
                H["track_err"].append(err)
        cx, cy = cfg.vehicle.center_of(e.x, e.y, e.yaw)
        ego_poly = box_corners(cx, cy, e.yaw, cfg.vehicle.length, cfg.vehicle.width)
        for a in world.actors:
            if abs(a.x - cx) < 15 and abs(a.y - cy) < 15:
                gap = polygon_distance(ego_poly, box_corners(a.x, a.y, a.yaw, a.length, a.width))
                fam = a.cls.family
                min_clear[fam] = min(min_clear.get(fam, np.inf), gap)
        if on_step is not None:
            on_step(world, stack)

        s, d = world.ego_frenet()
        in_goal_lane = scenario.goal_lane is None or world.map.lane_at(d) == scenario.goal_lane
        if s >= scenario.goal_s - 15.0 and e.v < 0.5:
            reached = in_goal_lane
            break
        if stop_on_collision and (world.collided or world.off_road):
            break

    t = np.array(H["t"])
    ex, ey = np.array(H["est_x"]) - H["x"], np.array(H["est_y"]) - H["y"]
    loc_err = np.hypot(ex, ey)
    yaw = np.array(H["yaw"])
    lon_err = ex * np.cos(yaw) + ey * np.sin(yaw)
    lat_err = -ex * np.sin(yaw) + ey * np.cos(yaw)
    ct = np.abs(np.array(H["cross_track"]))
    v, yr = np.array(H["v"]), np.array(H["yaw_rate"])
    jerk = np.abs(np.gradient(np.array(H["a"]), t)) if len(t) > 2 else np.zeros(1)
    end_s, _ = world.ego_frenet()
    finite_ttc = [x for x in H["ttc"] if np.isfinite(x)]
    # progress: reaching the goal, or at least averaging 35% of the limit (stops for people
    # and crossings included). A car that freezes at an empty crossing is not "safe", it is broken.
    avg = (end_s - start_s) / max(world.t, 1e-6)
    stalled = not reached and avg < 0.35 * scenario.map.speed_limit
    result = RunResult(
        scenario=scenario.name,
        success=not world.collided and not world.off_road and not stalled,
        stalled=stalled,
        collided=world.collided, off_road=world.off_road, reached_goal=reached,
        collision_speed=world.collision_speed, collision_with=world.collision_with,
        sim_time=float(world.t), distance=float(end_s - start_s),
        loc_rmse=float(np.sqrt(np.mean(loc_err**2))), loc_max_err=float(loc_err.max()),
        loc_lat_rmse=float(np.sqrt(np.mean(lat_err**2))), loc_lon_rmse=float(np.sqrt(np.mean(lon_err**2))),
        tracking_rmse=float(np.sqrt(np.mean(np.square(H["track_err"])))) if H["track_err"] else float("nan"),
        mean_cross_track=float(ct.mean()), max_cross_track=float(ct.max()),
        max_lat_accel=float(np.max(np.abs(v * yr))), max_jerk=float(np.percentile(jerk, 99)),
        min_ttc=float(min(finite_ttc)) if finite_ttc else float("inf"), min_clearance=min_clear,
        aeb_events=stack.safety.aeb_events, gnss_rejected=stack.localization.gnss_rejected,
        timing_ms=stack.timer.summary(), history=H)
    return result
