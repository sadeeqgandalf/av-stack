"""Motion planning: sampling-based Frenet lattice (Werling et al., ICRA 2010).

Each cycle:
  1. Pick a start state: stitch onto the previous plan (temporal consistency)
     unless the ego has drifted from it, then use the localization estimate.
  2. Sample end states: lateral {target lane, current lane} x offsets x horizons,
     longitudinal {velocity keeping targets} + {stop-at-point targets}.
  3. Build jerk-optimal quintic (lateral) / quartic or quintic (longitudinal)
     polynomials, convert to Cartesian.
  4. Reject candidates violating kinematic limits, road bounds or lane closures.
  5. Collision-check survivors against every predicted object (circle
     decomposition + prediction uncertainty margin).
  6. Choose the minimum-cost trajectory. No survivor -> emergency stop profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avstack.common.geometry import normalize_angle
from avstack.common.types import PredictedObject, Trajectory, VehicleState
from avstack.config import PlanningConfig, VehicleConfig
from avstack.map.hd_map import HDMap
from avstack.planning.behavior import BehaviorDecision
from avstack.planning.polynomials import ImmediateBrake, QuarticPolynomial, QuinticPolynomial, eval_with_hold


@dataclass
class FrenetPlan:
    t: np.ndarray            # absolute time
    s: np.ndarray
    s_d: np.ndarray
    s_dd: np.ndarray
    d: np.ndarray
    d_d: np.ndarray
    d_dd: np.ndarray


def _circles(x, y, yaw, length, width, n=5):
    """Approximate a box trajectory by n circles per timestep -> centres (K, n, 2), radii (K,).

    ``length``/``width`` may be per-timestep arrays, which is how prediction
    uncertainty is folded in: stretch along-track, widen across-track."""
    length = np.broadcast_to(np.asarray(length, float), x.shape)
    width = np.broadcast_to(np.asarray(width, float), x.shape)
    offs = (np.arange(n) - (n - 1) / 2)[None, :] * (length / n)[:, None]
    cx = x[:, None] + offs * np.cos(yaw)[:, None]
    cy = y[:, None] + offs * np.sin(yaw)[:, None]
    r = np.hypot(length / (2 * n), width / 2)
    return np.stack([cx, cy], axis=-1), r


class FrenetPlanner:
    def __init__(self, cfg: PlanningConfig, vcfg: VehicleConfig, hd_map: HDMap):
        self.cfg, self.vcfg, self.map = cfg, vcfg, hd_map
        self.horizon = max(cfg.horizon_times)
        self.ts = np.arange(0.0, self.horizon + 1e-9, cfg.dt)
        self.prev: FrenetPlan | None = None
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    def _start_state(self, t: float, ego: VehicleState):
        ref = self.map.reference
        s, d, s_d, d_d = ref.cartesian_to_frenet_state(ego.x, ego.y, ego.yaw, ego.v)
        if self.prev is not None and self.prev.t[0] <= t <= self.prev.t[-1]:
            p = self.prev
            ps, pd = np.interp(t, p.t, p.s), np.interp(t, p.t, p.d)
            if abs(ps - s) < 1.5 and abs(pd - d) < 0.3:
                # on track: continue the previous plan exactly (no jumps from estimate
                # noise); the controller is responsible for the small tracking error
                g = lambda arr: float(np.interp(t, p.t, arr))       # noqa: E731
                return ps, g(p.s_d), g(p.s_dd), pd, g(p.d_d), g(p.d_dd)
        return s, max(s_d, 0.0), ego.a, d, d_d, 0.0

    def plan(self, t: float, ego: VehicleState, beh: BehaviorDecision,
             predictions: list[PredictedObject]) -> tuple[Trajectory, list[Trajectory]]:
        start = self._start_state(t, ego)
        cur_lane = self.map.lane_at(start[3])
        lanes = {beh.target_lane, cur_lane}
        result = self._plan_lanes(t, start, lanes, beh, predictions)
        if result is None:
            # Evasive pass: the committed lanes are blocked (e.g. debris appeared before the
            # behaviour layer decided to change lanes). Adjacent lanes are still fully
            # collision-checked against their own traffic.
            evasive = {ln for ln in (cur_lane - 1, cur_lane + 1) if 0 <= ln < self.map.num_lanes} - lanes
            if evasive:
                result = self._plan_lanes(t, start, lanes | evasive, beh, predictions, evasive=True)
        if result is None:
            result = self._fallback(t, start, predictions)
        return result

    def _plan_lanes(self, t, start, lanes, beh, predictions, evasive: bool = False):
        c, m = self.cfg, self.map
        s0, sd0, sdd0, d0, dd0, ddd0 = start

        # lateral end states
        d_targets = []
        if sd0 < 2.5 and len(lanes) == 1:
            # crawling: lateral moves need large heading angles that swing the nose out; hold
            # the current line (kept inside the lane) and only settle toward the centre
            lane = next(iter(lanes))
            centre = m.lane_center_d(lane)
            hold = float(np.clip(d0, centre - 0.6, centre + 0.6))
            d_targets = [(hold, lane), (centre, lane)] if abs(hold - centre) > 0.05 else [(centre, lane)]
        else:
            offsets = tuple(c.lateral_offsets) + (tuple(c.nudge_offsets) if beh.nudge else ())
            for lane in lanes:
                for off in offsets:
                    d_targets.append((m.lane_center_d(lane) + off, lane))

        candidates, fplans, stats = [], [], {"generated": 0, "kinematic": 0, "collision": 0}
        conflicted = []                             # (clearance, conflict_speed, traj, fplan)
        for T in c.horizon_times:
            lon_options = []
            v_goal = float(np.clip(beh.target_speed, sd0 - 0.7 * c.max_decel * T, sd0 + 0.7 * c.max_accel * T))
            v_samples = {max(0.0, v_goal + dv) for dv in np.linspace(-6.0, 0.0, c.speed_samples)}
            # explicit braking family so a feasible "slow down hard" option always exists
            v_samples |= {max(0.0, sd0 - decel * T) for decel in (1.5, 3.0, 0.9 * c.max_decel)}
            for vT in sorted(v_samples):
                lon_options.append(("keep", QuarticPolynomial(s0, sd0, sdd0, vT, 0.0, T), vT))
            if beh.target_speed <= 0.0 and sd0 > 0.5:
                # behaviour demands a stop it can no longer make gently: offer an immediate brake
                lon_options.append(("brake", ImmediateBrake(s0, sd0, 0.9 * c.max_decel), 0.0))
            if beh.stop_s is not None and beh.stop_s > s0:
                for back in (0.0, 3.0):
                    sT = max(beh.stop_s - back, s0 + 0.1)
                    lon_options.append(("stop", QuinticPolynomial(s0, sd0, sdd0, sT, 0.0, 0.0, T), 0.0))

            low_speed = sd0 < 4.0
            for d_T, lane in d_targets:
                # at speed: d(t). At low speed (Werling's low-velocity mode): d(s), so the path's
                # curvature is bounded by geometry however slowly we roll, e.g. pulling out from
                # behind a stopped car. Several manoeuvre lengths are sampled.
                lat_variants = ([("t", QuinticPolynomial(d0, dd0, ddd0, d_T, 0.0, 0.0, T), T)] if not low_speed else
                                [("s", QuinticPolynomial(d0, dd0 / max(sd0, 1.0), 0.0, d_T, 0.0, 0.0, Ls), Ls)
                                 for Ls in (8.0, 12.0, 18.0)])
                for kind, lon, vT in lon_options:
                    if kind == "brake":
                        s, s_d, s_dd, s_ddd = (lon.eval(self.ts, k) for k in range(4))
                    else:
                        s, s_d, s_dd, s_ddd = eval_with_hold(lon, self.ts, T)
                    if kind == "stop":
                        s_d = np.where(self.ts > T, 0.0, s_d)
                        s = np.where(self.ts > T, lon.eval(T), s)
                    for mode, lat, span in lat_variants:
                        if mode == "t":
                            d, d_d, d_dd, d_ddd = eval_with_hold(lat, self.ts, span)
                        else:
                            q, q1, q2, _ = eval_with_hold(lat, np.maximum(s - s0, 0.0), span)
                            d, d_d = q, q1 * s_d
                            d_dd = q2 * s_d**2 + q1 * s_dd
                            d_ddd = np.gradient(d_dd, self.ts)
                        stats["generated"] += 1
                        fp = FrenetPlan(t + self.ts, s, s_d, s_dd, d, d_d, d_dd)
                        traj = self._to_cartesian(fp)
                        if not self._kinematically_ok(fp, traj) or not self._limits_ok(fp, beh, sd0):
                            stats["kinematic"] += 1
                            continue
                        ok, clearance, v_conflict = self._collision_free(traj, predictions)
                        if not ok:
                            stats["collision"] += 1
                            conflicted.append((clearance, v_conflict, traj, fp))
                            continue
                        traj.cost = self._cost(fp, T, d_T, lane, beh, clearance, s_ddd, d_ddd, kind)
                        if kind == "brake":
                            traj.cost = -1.0 + 0.01 * traj.cost     # urgent stop: comfort costs do not apply
                        traj.label = f"{kind} T={T:.0f} lane={lane} v={vT:.1f}"
                        candidates.append(traj)
                        fplans.append(fp)

        self.last_stats = stats
        self._conflicted = conflicted
        if not candidates:
            return None
        best = int(np.argmin([tr.cost for tr in candidates]))
        self.prev = fplans[best]
        if evasive:
            candidates[best].label = "EVASIVE " + candidates[best].label
        return candidates[best], candidates

    def _fallback(self, t, start, predictions):
        s0, sd0, _, d0, _, _ = start
        if self._conflicted:
            return self._minimal_risk(t, s0, sd0, d0, self._conflicted, predictions)
        fp = self._emergency_profile(t, s0, sd0, d0)
        self.prev = None           # never stitch onto an emergency: it would bias the next cycle
        traj = self._to_cartesian(fp)
        traj.label = "EMERGENCY_STOP"
        return traj, []

    def _minimal_risk(self, t, s0, sd0, d0, conflicted, predictions):
        """Nothing is provably clear. Two very different situations:

        * Only the *safety margins* are violated (best clearance > -0.3 m): a near-miss
          under conservative buffers. Hard braking on a highway would itself be a
          hazard, so take the candidate with the most clearance.
        * A real predicted overlap: contact may be unavoidable, so minimise impact
          energy, i.e. the lowest ego speed at the first conflict (the emergency
          profile competes too). This is what makes the car brake for a dog.
        """
        best_clear = max(c[0] for c in conflicted)
        if best_clear > -0.3:
            clearance, _, traj, fp = max(conflicted, key=lambda c: c[0])
            traj.label = f"LEAST_RISK clearance={clearance:.2f}"
            self.prev = fp
            return traj, []
        efp = self._emergency_profile(t, s0, sd0, d0)
        etraj = self._to_cartesian(efp)
        _, eclear, ev = self._collision_free(etraj, predictions)
        pool = conflicted + [(eclear, ev, etraj, None)]
        clearance, v_conf, traj, fp = min(pool, key=lambda c: (round(c[1], 1), -c[0]))
        traj.label = "EMERGENCY_STOP" if fp is None else f"MIN_RISK v_conflict={v_conf:.1f}"
        self.prev = fp                    # never stitch onto an emergency profile (fp is None)
        return traj, []

    # ------------------------------------------------------------------ #
    def _to_cartesian(self, fp: FrenetPlan) -> Trajectory:
        ref = self.map.reference
        x, y = ref.frenet_to_cartesian(fp.s, fp.d)
        _, _, ryaw, rk = ref.interpolate(fp.s)
        one_minus = np.maximum(1.0 - rk * fp.d, 0.1)
        yaw = normalize_angle(ryaw + np.arctan2(fp.d_d, np.maximum(fp.s_d, 0.5) * one_minus))
        v = np.hypot(fp.s_d * one_minus, fp.d_d)
        ds = np.maximum(np.hypot(np.gradient(x), np.gradient(y)), 1e-3)
        kappa = np.gradient(np.unwrap(yaw)) / ds
        kappa = np.where(v > 1.0, kappa, 0.0)
        return Trajectory(fp.t, x, y, yaw, v, fp.s_dd, kappa)

    def _kinematically_ok(self, fp: FrenetPlan, tr: Trajectory) -> bool:
        c, m, half_w = self.cfg, self.map, self.vcfg.width / 2
        if np.any(fp.s_d < -0.1) or np.any(tr.v > c.max_speed + 1.0):
            return False
        if np.any(fp.s_dd > c.max_accel + 0.5) or np.any(fp.s_dd < -c.max_decel - 0.5):
            return False
        if np.any(np.abs(tr.kappa) > c.max_curvature):
            return False
        if np.any(tr.v**2 * np.abs(tr.kappa) > c.max_lat_accel * 1.2):
            return False
        # keep the whole (rotated) footprint 30 cm inside the road edge: at low speed with a
        # large heading angle the front corner swings well past the rear axle's offset
        theta = np.arctan2(fp.d_d, np.maximum(fp.s_d, 0.5))
        rc = self.vcfg.rear_to_center
        d_centre = fp.d + rc * np.sin(theta)
        extent = half_w * np.abs(np.cos(theta)) + (self.vcfg.length / 2) * np.abs(np.sin(theta)) + 0.3
        viol = np.maximum(np.maximum(m.road_d_min - (d_centre - extent), (d_centre + extent) - m.road_d_max), 0.0)
        # a soft margin: we may *start* inside it (e.g. after an evasive nudge), but a plan
        # must never make it worse and must end clear of it; otherwise recovery is impossible
        if np.any(viol > viol[0] + 0.02) or viol[-1] > 0.0:
            return False
        # never drive into a closed lane
        for cl in m.closures:
            center = m.lane_center_d(cl.lane)
            inside = (fp.s + self.vcfg.length / 2 >= cl.s_start) & (fp.s <= cl.s_end) & \
                     (np.abs(fp.d - center) < m.lane_width / 2 + half_w - 0.2)
            if np.any(inside):
                return False
        return True

    @staticmethod
    def _limits_ok(fp: FrenetPlan, beh: BehaviorDecision, sd0: float) -> bool:
        """Location-bound speed limits (e.g. passing a hidden corner)."""
        for s_lim, v_max in beh.speed_limits or ():
            if fp.s[0] >= s_lim:
                # already inside the zone too fast (limit appeared late): only slowing down is allowed
                if np.any(fp.s_d > max(v_max, sd0) + 0.1):
                    return False
            elif np.any((fp.s >= s_lim) & (fp.s_d > v_max + 0.3)):
                return False
        return True

    def _collision_free(self, tr: Trajectory, preds: list[PredictedObject]) -> tuple[bool, float, float]:
        """-> (clear, minimum clearance [m], ego speed at the first conflict [m/s])."""
        c = self.cfg
        # trajectory is for the rear axle; the footprint is centred wheelbase/2 ahead
        cx, cy = self.vcfg.center_of(tr.x, tr.y, tr.yaw)
        ego_c, ego_r = _circles(cx, cy, tr.yaw, self.vcfg.length, self.vcfg.width)
        min_clear = np.inf
        first_conflict = len(tr.t)
        rel_t = tr.t - tr.t[0]
        ref = self.map.reference
        ego_s, ego_d = ref.project(cx[0], cy[0])
        # the lane our footprint centre is in: once past the divider we own that lane
        own_center = self.map.lane_center_d(self.map.lane_at(ego_d))
        for p in preds:
            if np.hypot(p.x[0] - cx[0], p.y[0] - cy[0]) > 120:
                continue
            ps, pd = ref.project(p.x[0], p.y[0])
            if ps < ego_s - self.vcfg.length and abs(pd - own_center) < 0.5 * self.map.lane_width:
                continue        # behind us in the lane we occupy: the follower must yield
            px, py = np.interp(rel_t, p.t, p.x), np.interp(rel_t, p.t, p.y)
            pyaw = np.interp(rel_t, p.t, np.unwrap(p.yaw))
            lon_std = np.interp(rel_t, p.t, p.lon_std)
            lat_std = np.interp(rel_t, p.t, p.lat_std)
            # 1-sigma inflated footprint: longer by the along-track uncertainty,
            # wider by the across-track uncertainty
            obj_c, obj_r = _circles(px, py, pyaw, p.length + 2 * lon_std, p.width + 2 * lat_std)
            dist = np.linalg.norm(ego_c[:, :, None, :] - obj_c[:, None, :, :], axis=-1).min(axis=(1, 2))
            clear = dist - ego_r - obj_r - c.family_margin.get(p.cls.family, c.safety_margin)
            min_clear = min(min_clear, float(clear.min()))
            bad = np.flatnonzero(clear < 0)
            if len(bad):
                first_conflict = min(first_conflict, int(bad[0]))
        v_conflict = float(tr.v[first_conflict]) if first_conflict < len(tr.t) else 0.0
        return bool(min_clear >= 0), min_clear, v_conflict

    def _cost(self, fp, T, d_T, lane, beh, clearance, s_ddd, d_ddd, kind) -> float:
        c, m = self.cfg, self.map
        j_lat = float(np.mean(d_ddd**2))
        j_lon = float(np.mean(s_ddd**2))
        v_ref = max(beh.target_speed, 1.0)
        speed_err = float(np.mean((fp.s_d[len(fp.s_d) // 2:] - beh.target_speed) ** 2)) / v_ref**2
        if kind == "stop":
            speed_err = 0.0 if beh.stop_s is not None and beh.target_speed < 2.0 else speed_err
        offset = ((d_T - m.lane_center_d(beh.target_lane) - beh.lateral_bias) / m.lane_width) ** 2
        prox = float(np.exp(-clearance)) if np.isfinite(clearance) else 0.0
        consistency = 0.0
        if self.prev is not None:
            consistency = 0.2 * (d_T - self.prev.d[-1]) ** 2
        return (c.w_jerk_lat * j_lat + c.w_jerk_lon * j_lon + c.w_time * 0.1 * T
                + c.w_offset * 5 * offset + c.w_speed * 10 * speed_err + c.w_proximity * prox
                + consistency)

    def _emergency_profile(self, t, s0, sd0, d0) -> FrenetPlan:
        a = -self.cfg.max_decel
        t_stop = sd0 / -a if sd0 > 0 else 0.0
        tt = np.minimum(self.ts, t_stop)
        s = s0 + sd0 * tt + 0.5 * a * tt**2
        s_d = np.maximum(sd0 + a * self.ts, 0.0)
        s_dd = np.where(self.ts < t_stop, a, 0.0)
        z = np.zeros_like(self.ts)
        return FrenetPlan(t + self.ts, s, s_d, s_dd, np.full_like(self.ts, d0), z, z.copy())
