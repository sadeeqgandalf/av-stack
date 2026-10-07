"""Record a closed-loop run into a compact log for the web replay viewer.

Captures, at a fixed rate: ground truth, raw sensor products (LiDAR points,
radar returns, camera renderings), every intermediate product of the stack
(localization, detections, open-vocab instances, tracks, predictions, the
candidate fan, the chosen plan, behaviour state), control, safety and timing.
Numbers are rounded to what the viewer can display, which keeps a minute of
driving to a few MB of JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from avstack.config import StackConfig
from avstack.sim.scenarios import Scenario

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _r(x, n=2):
    return round(float(x), n) if x is not None and math.isfinite(float(x)) else None


def _poly(xs, ys, step=1, n=2):
    return [[_r(x, n), _r(y, n)] for x, y in zip(xs[::step], ys[::step])]


class Recorder:
    def __init__(self, scenario: Scenario, cfg: StackConfig, every: int = 5, max_candidates: int = 14):
        self.scenario, self.cfg = scenario, cfg
        self.every, self.max_candidates = every, max_candidates
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._k = 0
        self._last = {"behavior": None, "aeb": False, "label": ""}
        self._lidar: list = []

    # ------------------------------------------------------------------ #
    def on_step(self, world, stack) -> None:
        dbg = stack.debug
        self._events(world.t, dbg)
        if dbg.lidar_world is not None and len(dbg.lidar_world):
            self._lidar = np.round(dbg.lidar_world[::1], 1).ravel().tolist()
        self._k += 1
        if self._k % self.every:
            return
        e, est = world.ego.state, dbg.ego_estimate
        ref = world.map.reference
        s, d = world.ego_frenet()
        traj = dbg.trajectory
        cands = sorted(dbg.candidates, key=lambda c: c.cost)[: self.max_candidates]
        costs = [c.cost for c in cands]
        cmin, cmax = (min(costs), max(costs)) if costs else (0.0, 1.0)
        cam_objs, cam_inst = [], []
        for f in dbg.camera_frames:
            for o in f.objects:
                cam_objs.append([f.camera, *[_r(v, 0) for v in o.box_xyxy], o.description, _r(o.visible_fraction)])
        for inst in stack.open_vocab.last_instances:
            cam_inst.append([inst.camera, *[_r(v, 0) for v in inst.box_xyxy], inst.prompt, _r(inst.score),
                             inst.description == "hallucination",
                             stack.open_vocab.prompt_class[inst.prompt].value])
        loc_err = math.hypot(est.x - e.x, est.y - e.y)
        lat_err = -(est.x - e.x) * math.sin(e.yaw) + (est.y - e.y) * math.cos(e.yaw)
        self.frames.append({
            "t": _r(world.t),
            "ego": [_r(e.x), _r(e.y), _r(e.yaw, 4), _r(e.v), _r(e.a), _r(e.steer, 4), _r(e.v * e.yaw_rate)],
            "est": [_r(est.x), _r(est.y), _r(est.yaw, 4)],
            "frenet": [_r(s, 1), _r(d)],
            "loc": [_r(loc_err, 3), _r(lat_err, 3)],
            "cmd": [_r(dbg.command.accel), _r(dbg.command.steer, 4), dbg.command.source] if dbg.command else None,
            "beh": dbg.behavior, "lane": dbg.target_lane, "vref": _r(dbg.target_speed),
            "stop": _r(dbg.stop_s, 1), "ttc": _r(min(dbg.min_ttc, 99.0)), "aeb": dbg.aeb_active,
            "plan": traj.label if traj is not None else "",
            "stats": dbg.planner_stats,
            "actors": [[a.id, _r(a.x), _r(a.y), _r(a.yaw, 3), _r(a.length), _r(a.width), _r(a.height),
                        a.cls.value, a.description] for a in world.actors],
            "tracks": [[tr.id, _r(tr.x), _r(tr.y), _r(tr.yaw, 3), _r(tr.length), _r(tr.width), tr.cls.value,
                        tr.label, _r(tr.speed, 1), "".join(sorted(x[0] for x in tr.sources)), tr.hits]
                       for tr in dbg.tracks],
            "preds": [[p.track_id, p.model, _poly(p.x, p.y, 2, 1), _r(p.lon_std[-1]), _r(p.lat_std[-1])]
                      for p in dbg.predictions],
            "traj": ([[_r(x), _r(y), _r(v, 1)] for x, y, v in zip(traj.x[::2], traj.y[::2], traj.v[::2])]
                     if traj is not None else []),
            "cands": [[_r((c.cost - cmin) / max(cmax - cmin, 1e-6)), _poly(c.x, c.y, 5, 1)] for c in cands],
            "lidar": self._lidar,
            "radar": [[_r(r.x, 1), _r(r.y, 1), _r(r.range_rate, 1)] for r in dbg.radar],
            "dets": [[_r(dd.x), _r(dd.y), dd.source, dd.label, dd.cls.value] for dd in dbg.detections],
            "cam_objs": cam_objs, "cam_inst": cam_inst,
            "timing": {k: _r(v[-1], 2) for k, v in stack.timer.samples.items() if v},
            "occ": [[_r(o.x), _r(o.y), bool(o.occluded), _r(o.v_safe, 1), o.track_id] for o in dbg.occlusion],
            "caps": {k: _r(v, 1) for k, v in dbg.caps.items()},
        })

    def _events(self, t: float, dbg) -> None:
        beh = (dbg.behavior or "").split(" / ")[0]
        if beh and beh != self._last["behavior"]:
            self.events.append({"t": _r(t), "kind": "behavior", "label": beh})
            self._last["behavior"] = beh
        if dbg.aeb_active and not self._last["aeb"]:
            self.events.append({"t": _r(t), "kind": "aeb", "label": f"AEB (TTC {dbg.min_ttc:.1f}s)"})
        self._last["aeb"] = dbg.aeb_active
        label = dbg.trajectory.label if dbg.trajectory is not None else ""
        risky = label.startswith(("EMERGENCY", "MIN_RISK", "EVASIVE", "LEAST_RISK"))
        if risky and not self._last["label"].startswith(label.split(" ")[0]):
            self.events.append({"t": _r(t), "kind": "fallback", "label": label.split(" ")[0]})
        self._last["label"] = label if risky else ""

    # ------------------------------------------------------------------ #
    def to_dict(self, result=None) -> dict:
        m = self.scenario.map
        lines = [{"style": style, "pts": _poly(p[:, 0], p[:, 1], 2, 2)} for style, p in m.boundaries()]
        closures = []
        for c in m.closures:
            ss = np.arange(c.s_start, c.s_end, 2.0)
            lo, hi = m.lane_center_d(c.lane) - m.lane_width / 2, m.lane_center_d(c.lane) + m.lane_width / 2
            x0, y0 = m.reference.frenet_to_cartesian(ss, np.full_like(ss, lo))
            x1, y1 = m.reference.frenet_to_cartesian(ss[::-1], np.full_like(ss, hi))
            closures.append(_poly(np.r_[x0, x1], np.r_[y0, y1]))
        crosswalks = []
        for cw in m.crosswalks:
            ss = np.array([cw.s_start, cw.s_end, cw.s_end, cw.s_start])
            dd = np.array([m.road_d_min, m.road_d_min, m.road_d_max, m.road_d_max])
            x, y = m.reference.frenet_to_cartesian(ss, dd)
            _, _, yaw, _ = m.reference.interpolate(cw.s)
            crosswalks.append({"poly": _poly(x, y), "yaw": _r(yaw, 4), "s": cw.s, "width": cw.width,
                               "d": [m.road_d_min, m.road_d_max]})
        v = self.cfg.vehicle
        summary = None
        if result is not None:
            summary = {k: getattr(result, k) for k in (
                "success", "collided", "off_road", "reached_goal", "sim_time", "distance", "loc_rmse",
                "loc_lat_rmse", "tracking_rmse", "mean_cross_track", "max_lat_accel", "max_jerk", "min_ttc",
                "aeb_events", "min_clearance")}
            summary["timing_ms"] = {k: _r(m_, 2) for k, (m_, _) in result.timing_ms.items()}
        return {
            "meta": {"scenario": self.scenario.name, "description": self.scenario.description,
                     "tags": self.scenario.tags, "dt": self.cfg.sim_dt * self.every,
                     "goal_s": self.scenario.goal_s, "goal_lane": self.scenario.goal_lane,
                     "speed_limit": m.speed_limit, "vocabulary": list(self.cfg.perception.open_vocab.vocabulary)},
            "vehicle": {"length": v.length, "width": v.width, "wheelbase": v.wheelbase,
                        "rear_to_center": v.rear_to_center},
            "cameras": [asdict(c) for c in self.cfg.sensors.cameras],
            "map": {"lines": lines, "closures": closures, "crosswalks": crosswalks, "num_lanes": m.num_lanes,
                    "lane_width": m.lane_width},
            "events": self.events, "summary": summary, "frames": self.frames,
        }


def export_html(data: dict, out_path: str | Path) -> Path:
    """Write a self-contained viewer: template + the log embedded as JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"), default=lambda o: None).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__AVSTACK_DATA__*/null", payload)
    out.write_text(html)
    return out
