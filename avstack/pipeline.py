"""The AV stack: wires every module together with its own update rate.

    sensors ─┬─> localization (EKF, every tick) ───────────────┐
             ├─> LiDAR detector ─┐                            │ ego pose
             └─> camera detector ┴─> tracker (fusion) ────────┤
                                        │ tracks              │
                                        v                     v
                            prediction ─> behaviour (FSM + routing)
                                        │                     │
                                        v                     v
                                 Frenet local planner (10 Hz) ─> trajectory
                                                              │
                                        controller (50 Hz) <──┘
                                              │ command
                                        safety monitor (AEB/watchdog) ─> actuators

The stack only ever consumes a SensorBundle and produces a ControlCommand; it
has no access to simulator ground truth. Swap the simulator for a real
vehicle interface (or a ROS 2 bridge) and nothing in here changes.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from avstack.common.timing import RateGate
from avstack.common.types import ControlCommand, DebugInfo, SensorBundle, VehicleState
from avstack.config import StackConfig
from avstack.control.controller import TrajectoryController
from avstack.localization.ekf import LocalizationEKF
from avstack.map.hd_map import HDMap
from avstack.map.routing import Router
from avstack.perception.open_vocab.detector import OpenVocabPerception
from avstack.perception.radar_detector import RadarDetector
from avstack.perception.lidar_detector import LidarDetector
from avstack.planning.behavior import BehaviorPlanner
from avstack.planning.local_planner import FrenetPlanner
from avstack.prediction.predictor import LanePredictor
from avstack.safety.monitor import SafetyMonitor
from avstack.tracking.tracker import MultiObjectTracker


class _Timer:
    """Per-module wall-clock profiling (mean/max ms)."""

    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.samples[name].append((time.perf_counter() - self.t0) * 1e3)
        return _Ctx()

    def summary(self) -> dict[str, tuple[float, float]]:
        return {k: (float(np.mean(v)), float(np.max(v))) for k, v in self.samples.items()}


class AVStack:
    def __init__(self, cfg: StackConfig, hd_map: HDMap, initial_guess: VehicleState,
                 goal_s: float, goal_lane: int | None = None):
        self.cfg, self.map = cfg, hd_map
        self.localization = LocalizationEKF(cfg.localization, cfg.sensors, initial_guess, hd_map)
        self.lidar_detector = LidarDetector(cfg.perception, hd_map)
        self.open_vocab = OpenVocabPerception(cfg.perception, cfg.sensors, hd_map)
        self.radar_detector = RadarDetector(cfg.perception, cfg.sensors, hd_map)
        self.tracker = MultiObjectTracker(cfg.tracking, velocity_prior=self._velocity_prior,
                                          lane_heading=self._lane_heading)
        self.predictor = LanePredictor(cfg.prediction, hd_map)
        self.router = Router(hd_map, cfg.planning.routing_segment, cfg.planning.routing_lane_change_cost)
        self.behavior = BehaviorPlanner(cfg.planning, cfg.vehicle, hd_map, self.router, goal_s, goal_lane)
        self.planner = FrenetPlanner(cfg.planning, cfg.vehicle, hd_map)
        self.controller = TrajectoryController(cfg.control, cfg.vehicle)
        self.safety = SafetyMonitor(cfg.safety, cfg.vehicle)

        self.plan_gate = RateGate(cfg.planning.rate)
        self._last_radar: tuple[float, list] = (-1.0, [])
        self.trajectory = None
        self.plan_time: float | None = None
        self.last_t: float | None = None
        self.timer = _Timer()
        self.debug = DebugInfo()

    def _lane_heading(self, x: float, y: float) -> float | None:
        s, d = self.map.reference.project(x, y)
        return float(self.map.reference.interpolate(s)[2]) if self.map.on_road(d, 2.0) else None

    def _velocity_prior(self, x: float, y: float, cls, length: float = 4.5) -> float:
        """Speed prior for a new track: vehicles on the road move with traffic,
        everything else (pedestrians, cones, debris, small unknown blobs) starts at rest."""
        if cls.family == "vehicle" or (cls.family == "unknown" and length >= 3.0):
            return self.localization.state.v
        return 0.0

    def step(self, sensors: SensorBundle) -> ControlCommand:
        t = sensors.t
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        dbg = self.debug
        dbg.t = t

        # 1) localization --------------------------------------------------
        with self.timer("localization"):
            if sensors.imu is not None and dt > 0:
                self.localization.predict(sensors.imu, dt)
            if sensors.odom is not None:
                self.localization.update_odom(sensors.odom)
            if sensors.gnss is not None:
                self.localization.update_gnss(sensors.gnss)
            if sensors.lane is not None:
                self.localization.update_lane(sensors.lane)
            ego = self.localization.state
        dbg.ego_estimate, dbg.ego_cov = ego, self.localization.P.copy()

        # 2) perception + 3) tracking/fusion --------------------------------
        # cameras are hardware-triggered with the LiDAR sweep, so both arrive on the same tick
        dets = None
        with self.timer("perception"):
            radar_dets = self.radar_detector.detect(sensors.radar, ego) if sensors.radar else None
            if radar_dets is not None:
                self._last_radar = (t, radar_dets)
                dbg.radar = radar_dets
            if sensors.cameras:
                dbg.camera_frames = sensors.cameras
            lidar_dets = self.lidar_detector.detect(sensors.lidar, ego) if sensors.lidar else None
            if sensors.cameras:
                recent_radar = self._last_radar[1] if t - self._last_radar[0] < 0.06 else []
                dets = self.open_vocab.process(sensors.cameras, lidar_dets, ego, recent_radar)
            elif lidar_dets is not None:
                dets = lidar_dets
        with self.timer("tracking"):
            self.tracker.predict(t)
            if dets is not None:
                self.tracker.update(t, dets, "lidar" if lidar_dets is not None else "camera",
                                    sensor_xy=np.array([ego.x, ego.y]))
            if radar_dets:
                self.tracker.update(t, radar_dets, "radar")
            if lidar_dets is not None:
                dbg.lidar_world = self.lidar_detector.last_world_points
            tracks = self.tracker.confirmed_tracks()
        if dets is not None:
            dbg.detections = dets
        dbg.tracks = tracks

        # 4) prediction, 5) behaviour + routing, 6) motion planning ---------
        if self.plan_gate.ready(t):
            with self.timer("prediction"):
                preds = self.predictor.predict(tracks)
            with self.timer("behavior"):
                centre = np.array(self.cfg.vehicle.center_of(ego.x, ego.y, ego.yaw))
                cs, cd = self.map.reference.project(*centre)
                decision = self.behavior.update(t, cs, cd, ego.v, tracks, ego_xy=centre)
            with self.timer("planning"):
                self.trajectory, cands = self.planner.plan(t, ego, decision, preds)
                self.plan_time = t
            dbg.predictions, dbg.candidates = preds, cands
            dbg.behavior, dbg.target_lane = decision.state, decision.target_lane
            dbg.target_speed, dbg.stop_s = decision.target_speed, decision.stop_s
            dbg.occlusion, dbg.caps = decision.occlusion or [], decision.caps or {}
            dbg.planner_stats = dict(self.planner.last_stats)
            dbg.route = self.behavior.route
            if self.trajectory.label.startswith(("EMERGENCY_STOP", "MIN_RISK")):
                dbg.behavior += " / EMERGENCY_STOP"
        dbg.trajectory = self.trajectory

        # 7) control, 8) safety ---------------------------------------------
        with self.timer("control"):
            cmd = (self.controller.compute(t, ego, self.trajectory) if self.trajectory is not None
                   else ControlCommand(accel=-3.0, source="fallback"))
        with self.timer("safety"):
            cmd = self.safety.check(t, ego, tracks, cmd, self.plan_time)
        dbg.command, dbg.aeb_active, dbg.min_ttc = cmd, self.safety.aeb_active, self.safety.min_ttc
        return cmd
