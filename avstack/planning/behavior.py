"""Behaviour layer: tactical decisions as a finite state machine.

States
  LANE_KEEP       cruise in the current lane at the (curvature-limited) speed
  FOLLOW          a slower lead is ahead: ACC-style gap keeping
  LANE_CHANGE     committed move to an adjacent lane (discretionary or mandatory)
  PREPARE_MERGE   route needs a lane change but the gap is unsafe: slow and wait
  STOP            reaching the goal / end of a closed lane: stop at a point
  YIELD           a pedestrian / cyclist / animal is crossing our path: stop short of
                  it and wait until it has cleared; never "overtake" a crossing VRU
  CROSSWALK       stop at a marked crossing: people in it or waiting at its kerb, or a
                  vehicle stopped beside us at it (the "multiple threat": it may hide
                  someone crossing). Proceed only after a full stop with the crossing clear.

Speed caps layered on top of any state:
  occlusion       parked / stopped vehicles near our path hide space where a person could
                  step out; see planning/occlusion.py for the bound.

Outputs a BehaviorDecision that parameterises the local (motion) planner.
The motion planner still collision-checks everything, so a bad decision
degrades comfort, not safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avstack.common.types import Track
from avstack.config import PlanningConfig, VehicleConfig
from avstack.map.hd_map import Crosswalk, HDMap
from avstack.map.routing import Router, RouteNode
from avstack.planning.occlusion import OcclusionReasoner, OcclusionRisk

MANDATORY_HORIZON = 250.0      # start caring about a required lane change this far out


@dataclass
class FrenetObject:
    id: int
    s: float
    d: float
    s_dot: float
    d_dot: float
    lane: int
    length: float
    family: str = "vehicle"
    width: float = 1.9


@dataclass
class BehaviorDecision:
    state: str
    target_lane: int
    target_speed: float
    stop_s: float | None = None
    lead: FrenetObject | None = None
    nudge: bool = False            # small obstacle near our path: sample in-lane nudges
    lateral_bias: float = 0.0      # preferred offset within the lane (away from occluders)
    occlusion: list[OcclusionRisk] | None = None
    caps: dict | None = None       # every active speed cap, by reason (for debugging / HUD)
    # hard, location-bound limits the motion planner must satisfy: at rear-axle s >= s_lim,
    # speed <= v_max. A target speed is only a preference; safety needs a constraint.
    speed_limits: list[tuple[float, float]] | None = None


class BehaviorPlanner:
    def __init__(self, cfg: PlanningConfig, vcfg: VehicleConfig, hd_map: HDMap, router: Router,
                 goal_s: float, goal_lane: int | None = None):
        self.cfg, self.vcfg, self.map, self.router = cfg, vcfg, hd_map, router
        self.goal_s, self.goal_lane = goal_s, goal_lane
        self.state = "LANE_KEEP"
        self.target_lane: int | None = None
        self.last_lc_time = -np.inf                 # time the last lane change *completed*
        self.t_start: float | None = None
        self.route: list[RouteNode] = []
        self._route_key = None
        self.occlusion = OcclusionReasoner(cfg, vcfg, hd_map)
        self._cw: dict[float, dict] = {}            # per-crosswalk state (stop timers)

    # ------------------------------------------------------------------ #
    def to_frenet(self, tracks: list[Track]) -> list[FrenetObject]:
        out = []
        for tr in tracks:
            heading = np.arctan2(tr.vy, tr.vx) if tr.speed > 1.0 else tr.yaw
            s, d, s_dot, d_dot = self.map.reference.cartesian_to_frenet_state(tr.x, tr.y, heading, tr.speed)
            out.append(FrenetObject(tr.id, s, d, s_dot, d_dot, self.map.lane_at(d), tr.length,
                                    tr.cls.family, tr.width))
        return out

    def _lead_in_lane(self, objs, lane, ego_s, max_dist=150.0):
        lw = self.map.lane_width
        center = self.map.lane_center_d(lane)
        cands = [o for o in objs if abs(o.d - center) < 0.6 * lw and 0 < o.s - ego_s < max_dist]
        return min(cands, key=lambda o: o.s, default=None)

    def gap_is_safe(self, objs, lane, ego_s, ego_v) -> bool:
        c, lw = self.cfg, self.map.lane_width
        center = self.map.lane_center_d(lane)
        for o in objs:
            # objects in the lane, or moving into it
            if abs(o.d - center) > 0.75 * lw and abs(o.d + 2.0 * o.d_dot - center) > 0.6 * lw:
                continue
            gap = o.s - ego_s
            half = (o.length + self.vcfg.length) / 2
            if gap >= 0 and gap - half < max(c.lane_change_front_gap * 0.5, (ego_v - o.s_dot) * 3.0, 5.0):
                return False
            if gap < 0 and -gap - half < max(c.lane_change_rear_gap, (o.s_dot - ego_v) * 3.0):
                return False
        return True

    def _usable(self, lane: int, s: float) -> float:
        """How far ahead a lane stays usable (closures, goal lane)."""
        free = self.map.free_distance(lane, s)
        if self.goal_lane is not None and lane != self.goal_lane:
            free = min(free, max(self.goal_s - s, 0.0))
        return free

    def _crossing_vru(self, objs, ego_s: float, ego_d: float, ego_v: float) -> FrenetObject | None:
        """Nearest VRU ahead that is on, or walking onto, the roadway in front of us."""
        m, best = self.map, None
        for o in objs:
            ahead = o.s - ego_s
            if o.family != "vru" or not 0.0 < ahead < max(40.0, 3.0 * ego_v):
                continue
            # crossing = predominantly lateral motion (a cyclist riding along with a bit of
            # tracker jitter is not crossing)
            lateral_mover = abs(o.d_dot) > 0.3 and abs(o.d_dot) > 0.5 * abs(o.s_dot)
            if not lateral_mover:
                continue
            on_road = m.on_road(o.d, 0.5)
            toward = (ego_d - o.d) * o.d_dot > 0
            reaches = toward and abs(ego_d - o.d) - m.lane_width / 2 < 5.0 * abs(o.d_dot)
            crossing = on_road
            if (reaches or crossing) and (best is None or ahead < best.s - ego_s):
                best = o
        return best

    def _discretionary_ok(self, lane: int, s: float, v: float) -> bool:
        """May we *choose* to move into ``lane``? Not if it closes soon, and not if it
        takes us further from a required goal lane than we could recover from."""
        if self._usable(lane, s) < max(MANDATORY_HORIZON, 10.0 * v):
            return False
        if self.goal_lane is not None:
            needed = abs(lane - self.goal_lane)
            if self.goal_s - s < (needed + 1) * max(MANDATORY_HORIZON, 10.0 * v):
                return False
        return True

    def _curvature_speed(self, s: float, v: float) -> float:
        ss = np.linspace(s, s + max(30.0, 3.0 * v), 20)
        kmax = float(np.max(np.abs(self.map.reference.interpolate(ss)[3])))
        return float(np.sqrt(self.cfg.max_lat_accel * 0.8 / max(kmax, 1e-4)))

    def _acc_speed(self, lead: FrenetObject, ego_s: float, ego_v: float, v_des: float) -> float:
        c = self.cfg
        gap = lead.s - ego_s - (lead.length + self.vcfg.length) / 2
        desired = c.min_gap + c.time_gap * ego_v
        return float(np.clip(lead.s_dot + 0.35 * (gap - desired), 0.0, v_des))

    # ------------------------------------------------------------------ #
    def update(self, t: float, ego_s: float, ego_d: float, ego_v: float, tracks: list[Track],
               ego_xy: np.ndarray | None = None) -> BehaviorDecision:
        c, m = self.cfg, self.map
        objs = self.to_frenet(tracks)
        cur_lane = m.lane_at(ego_d)
        if self.target_lane is None:
            self.target_lane = cur_lane
        if self.t_start is None:
            self.t_start = t

        # re-route when our lane (the start of the route) changes
        key = (cur_lane, int(ego_s // self.router.seg_len))
        if key != self._route_key:
            self.route = self.router.plan(ego_s, cur_lane, self.goal_s, self.goal_lane)
            self._route_key = key

        v_des = min(m.speed_limit, c.max_speed, self._curvature_speed(ego_s, ego_v))
        stop_s = None

        # ---- yield to crossing vulnerable road users (highest priority) ---- #
        crossing = self._crossing_vru(objs, ego_s, ego_d, ego_v)
        if crossing is None and self.state in ("YIELD", "CROSSWALK"):
            self.state = "LANE_KEEP"                 # the path is clear again: re-decide normally
        if crossing is not None:
            if self.state != "LANE_CHANGE" or abs(ego_d - m.lane_center_d(cur_lane)) < 1.0:
                self.target_lane = cur_lane          # stay put: swerving toward people is worse
            self.state = "YIELD"
            stop_s = crossing.s - (crossing.length + self.vcfg.length) / 2 - 4.0

        # ---- finish an in-progress lane change -------------------------- #
        if self.state == "LANE_CHANGE":
            if abs(ego_d - m.lane_center_d(self.target_lane)) < 0.5:
                self.state = "LANE_KEEP"
                self.last_lc_time = t
            elif cur_lane != self.target_lane and not self.gap_is_safe(objs, self.target_lane, ego_s, ego_v) \
                    and abs(ego_d - m.lane_center_d(cur_lane)) < 0.8:
                self.target_lane = cur_lane                                    # abort early
                self.state = "LANE_KEEP"
                self.last_lc_time = t

        if self.state not in ("LANE_CHANGE", "YIELD"):
            self.target_lane = cur_lane
            usable = self._usable(cur_lane, ego_s)
            route_lane = self.router.lane_at(self.route, ego_s + min(usable, 1e4) + 1.0)
            n_changes = abs(route_lane - cur_lane) if route_lane is not None else 0
            # each required lane change needs its own window, and more room at speed
            horizon = n_changes * max(MANDATORY_HORIZON, 10.0 * ego_v)
            mandatory = n_changes > 0 and usable < horizon
            lead = self._lead_in_lane(objs, cur_lane, ego_s)
            # discretionary manoeuvres need: settled perception, cooldown, not about to stop
            cooled = (t - self.last_lc_time > c.lane_change_cooldown
                      and t - self.t_start > c.perception_warmup
                      and self.goal_s - ego_s > 150.0)

            if mandatory:
                nxt = cur_lane + int(np.sign(route_lane - cur_lane))
                if self.gap_is_safe(objs, nxt, ego_s, ego_v) and t - self.t_start > c.perception_warmup:
                    self.state, self.target_lane = "LANE_CHANGE", nxt
                else:
                    self.state = "PREPARE_MERGE"
                    stop_s = ego_s + max(usable - 5.0, 0.0)
                    v_des = min(v_des, max(ego_v - 2.0, 8.0))                 # open up a gap
            elif lead is not None and lead.s_dot < c.overtake_speed_ratio * v_des \
                    and lead.s - ego_s < max(70.0, 4.0 * ego_v) and cooled:
                for nxt in ((cur_lane + 1, cur_lane - 1) if c.allow_right_pass else (cur_lane + 1,)):
                    if not 0 <= nxt < m.num_lanes or not self._discretionary_ok(nxt, ego_s, ego_v):
                        continue
                    nlead = self._lead_in_lane(objs, nxt, ego_s)
                    better = nlead is None or nlead.s_dot > lead.s_dot + 2.0 or nlead.s - ego_s > lead.s - ego_s + 30
                    if better and self.gap_is_safe(objs, nxt, ego_s, ego_v):
                        self.state, self.target_lane = "LANE_CHANGE", nxt
                        break
                else:
                    self.state = "FOLLOW"
            elif cur_lane > 0 and cooled and (lead is None or lead.s - ego_s > 80) and \
                    self._discretionary_ok(cur_lane - 1, ego_s, ego_v) and \
                    self._lead_in_lane(objs, cur_lane - 1, ego_s, 80.0) is None and \
                    self.gap_is_safe(objs, cur_lane - 1, ego_s, ego_v):
                # keep right when the right lane is clear
                self.state, self.target_lane = "LANE_CHANGE", cur_lane - 1
            else:
                self.state = "FOLLOW" if lead is not None and lead.s - ego_s < 100 else "LANE_KEEP"

        # ---- speed target: respect the lead in the target lane AND, until we
        # have actually left it, the lead in the lane we are still in --------- #
        lanes = {self.target_lane}
        leaving = abs(ego_d - m.lane_center_d(self.target_lane)) > 0.5 * m.lane_width
        # while still in the old lane, respect its lead too, except when edging around a stopped
        # obstacle at low speed: then that obstacle would pin the speed to 0 and the car could
        # never pull out (the motion planner's collision check still keeps us off it)
        if leaving and not (self.state == "LANE_CHANGE" and ego_v < 5.0):
            lanes.add(cur_lane)
        lead = None
        for ln in lanes:
            ld = self._lead_in_lane(objs, ln, ego_s)
            if ld is None:
                continue
            v_des = min(v_des, self._acc_speed(ld, ego_s, ego_v, v_des))
            if ln == self.target_lane:
                lead = ld
            if ld.s_dot < 1.0:
                # a stopped obstacle in either lane: stopping behind it must always be an option
                s_stop = ld.s - (ld.length + self.vcfg.length) / 2 - c.min_gap
                stop_s = s_stop if stop_s is None else min(stop_s, s_stop)
        # ---- vulnerable road users near our path: slow down, give room ----- #
        nudge = False
        ego_lane_d = m.lane_center_d(self.target_lane)
        for o in objs:
            ahead = o.s - ego_s
            if not -5.0 < ahead < 60.0:
                continue
            lateral = abs(o.d - ego_lane_d)
            heading_to_road = (ego_lane_d - o.d) * o.d_dot > 0 and abs(o.d_dot) > 0.3
            if o.family == "vru" and (lateral < m.lane_width * 1.6 or
                                      (heading_to_road and m.on_road(o.d, 4.0))):
                # "cover the brake": people near or stepping toward the road, cyclists alongside
                v_des = min(v_des, max(o.s_dot + 4.0, 7.0))
            if o.family in ("vru", "static", "unknown") and o.width < 2.5 and lateral < m.lane_width:
                nudge = True

        caps: dict[str, float] = {}
        # ---- crosswalks ------------------------------------------------------ #
        if c.crosswalk_rules:
            cw_stop, cw_cap, why = self._crosswalk_rules(t, objs, ego_s, ego_d, ego_v, cur_lane)
            if cw_stop is not None:
                stop_s = cw_stop if stop_s is None else min(stop_s, cw_stop)
                if self.state not in ("LANE_CHANGE", "YIELD"):
                    self.state = "CROSSWALK"
                caps["crosswalk: " + why] = 0.0              # reported: we are stopping for it
            elif np.isfinite(cw_cap):
                caps["crosswalk: " + why] = cw_cap
            v_des = min(v_des, cw_cap)
        # ---- occlusions ------------------------------------------------------ #
        risks, bias, limits = [], 0.0, []
        if c.occlusion_aware and ego_xy is not None:
            risks = self.occlusion.assess(ego_xy, ego_s, ego_d, m.lane_center_d(self.target_lane), tracks)
            live = [r for r in risks if r.occluded]
            # rear-axle s at which the front bumper reaches the corner (2 m early, for margin)
            front_offset = self.vcfg.rear_to_center + self.vcfg.length / 2
            limits = [(r.s - front_offset - 2.0, r.v_safe) for r in live]
            if live:
                worst = min(live, key=lambda r: r.v_cap)
                if worst.v_cap < v_des:
                    caps[f"occlusion #{worst.track_id}"] = worst.v_cap
                v_des = min(v_des, worst.v_cap)
                # lean away from the nearest hidden corner (the planner checks it is free)
                near = min(live, key=lambda r: r.s)
                centre = m.lane_center_d(self.target_lane)
                room = self.vcfg.width / 2 + 0.3
                bias = float(np.clip(-near.side * 0.6, m.road_d_min + room - centre, m.road_d_max - room - centre))

        # route end
        if self.goal_s - ego_s < max(80.0, ego_v**2 / (2 * 1.5)):
            self.state = "STOP"
            stop_s = min(stop_s if stop_s is not None else np.inf, self.goal_s)
        if stop_s is not None:
            # speed that lets us stop at stop_s with a comfortable 1.5 m/s^2
            v_stop = float(np.sqrt(2.0 * 1.5 * max(stop_s - ego_s - 2.0, 0.0)))
            if self.state in ("YIELD", "CROSSWALK") and ego_v > v_stop + 1.0:
                v_des = 0.0          # a person is involved and a gentle stop no longer fits: brake now
            else:
                v_des = min(v_des, v_stop)
        return BehaviorDecision(self.state, self.target_lane, v_des, stop_s, lead, nudge,
                                lateral_bias=bias, occlusion=risks, caps=caps, speed_limits=limits)

    # ------------------------------------------------------------------ #
    def _crosswalk_rules(self, t, objs, ego_s, ego_d, ego_v, cur_lane):
        """-> (stop_s or None, speed cap, reason) for the next marked crossing."""
        m, c, L = self.map, self.cfg, self.vcfg.length
        front = ego_s + L / 2
        for cw in m.crosswalks_ahead(front, 60.0):
            st = self._cw.setdefault(cw.s, {"stopped_since": None, "released": False})
            if front > cw.s_start - 0.5:
                if front < cw.s_end + L:              # already in it: clear it, gently
                    return None, c.crosswalk_creep_speed + 2.0 if st["released"] else np.inf, "clearing"
                continue
            stop_line = cw.s_start - 1.5 - L / 2      # ego-centre s at which to stop
            lane_c = m.lane_center_d(cur_lane)
            zone = [o for o in objs if o.family == "vru" and cw.s_start - 2.0 <= o.s <= cw.s_end + 2.0]
            in_road = [o for o in zone if m.on_road(o.d, 0.3) and not (
                (o.d - ego_d) * o.d_dot > 0 and abs(o.d - lane_c) > m.lane_width)]      # not walking away
            at_kerb = [o for o in zone if not m.on_road(o.d, 0.3) and m.on_road(o.d, c.crosswalk_curb_zone)
                       and (abs(o.d_dot) < 0.5 or (lane_c - o.d) * o.d_dot > 0)]
            threat = [o for o in objs if o.family == "vehicle" and o.lane != cur_lane
                      and abs(o.lane - cur_lane) == 1 and o.s_dot < 1.5
                      and cw.s_start - 12.0 <= o.s + o.length / 2 <= cw.s_start + 1.0]
            stopped_here = ego_v < 0.3 and abs(ego_s - stop_line) < 6.0
            if stopped_here and st["stopped_since"] is None:
                st["stopped_since"] = t
            waited = t - st["stopped_since"] if st["stopped_since"] is not None else 0.0
            if in_road:
                st["released"] = False
                return stop_line, np.inf, "people crossing"
            if at_kerb and waited < c.crosswalk_wait_timeout:
                return stop_line, np.inf, "people at kerb"
            if threat and not st["released"]:
                if waited >= 1.0:
                    st["released"] = True             # full stop made, crossing looks clear
                else:
                    return stop_line, np.inf, "vehicle stopped at crossing"
            if st["released"] or at_kerb:
                return None, c.crosswalk_creep_speed, "creep through"
            # approaching a crossing: be able to stop at its line comfortably
            room = max(stop_line - ego_s, 0.0)
            return None, float(np.sqrt(2.0 * 2.0 * room + 8.0**2)), "approach"
        return None, np.inf, ""
