"""Monte Carlo evaluation: every scenario x many seeds, in parallel.

Each seed changes every stochastic element: sensor noise, GNSS multipath jumps,
LiDAR dropout, radar clutter, open-vocab misses/hallucinations, and the
initial localization error. A stack that passes one seed proves little; the
pass *rate* and the tails of the metric distributions are what matter.

    PYTHONPATH=. .venv/bin/python scripts/evaluate.py --seeds 10
    PYTHONPATH=. .venv/bin/python scripts/evaluate.py --scenarios urban_vru road_debris --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from avstack.config import StackConfig
from avstack.runner import run
from avstack.sim.scenarios import SCENARIOS


def _one(args):
    name, seed, ablate = args
    cfg = StackConfig()
    if ablate:          # what do occlusion reasoning + crosswalk rules actually buy?
        cfg.planning.occlusion_aware = False
        cfg.planning.crosswalk_rules = False
    r = run(SCENARIOS[name](), cfg, seed=seed)
    return {
        "scenario": name, "seed": seed, "success": r.success, "collided": r.collided, "off_road": r.off_road,
        "collision_speed": r.collision_speed, "collision_with": r.collision_with,
        "reached_goal": r.reached_goal, "aeb_events": r.aeb_events, "min_ttc": r.min_ttc,
        "min_clearance": r.min_clearance, "loc_lat_rmse": r.loc_lat_rmse, "loc_rmse": r.loc_rmse,
        "tracking_rmse": r.tracking_rmse, "max_lat_accel": r.max_lat_accel, "max_jerk": r.max_jerk,
        "emergency_ticks": sum("EMERGENCY" in b or "MIN_RISK" in b for b in r.history["behavior"]),
        "distance": r.distance, "sim_time": r.sim_time,
        "planning_ms": r.timing_ms.get("planning", (0, 0))[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/evaluation.json")
    ap.add_argument("--ablate", action="store_true", help="disable occlusion reasoning and crosswalk rules")
    a = ap.parse_args()

    jobs = [(s, k, a.ablate) for s in a.scenarios for k in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, jobs))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, default=float)

    print(f"{len(rows)} runs in {time.time() - t0:.0f}s  ({a.workers} workers)\n")
    hdr = (f"{'scenario':18s} {'pass':>6s} {'coll':>5s} {'AEB':>4s} {'emerg':>6s} {'minTTC p5':>9s} "
           f"{'VRU clr':>8s} {'latLoc':>7s} {'trkRMSE':>8s} {'ay max':>7s} {'jerk p95':>8s}")
    print(hdr + "\n" + "-" * len(hdr))
    for s in a.scenarios:
        R = [r for r in rows if r["scenario"] == s]
        vru = [r["min_clearance"].get("vru") for r in R if "vru" in r["min_clearance"]]
        ttc = [r["min_ttc"] for r in R if np.isfinite(r["min_ttc"])]
        print(f"{s:18s} {np.mean([r['success'] for r in R]) * 100:5.0f}% "
              f"{sum(r['collided'] for r in R):5d} {sum(r['aeb_events'] for r in R):4d} "
              f"{np.mean([r['emergency_ticks'] > 0 for r in R]) * 100:5.0f}% "
              f"{np.percentile(ttc, 5) if ttc else float('inf'):9.2f} "
              f"{(min(vru) if vru else float('nan')):8.2f} "
              f"{np.mean([r['loc_lat_rmse'] for r in R]):7.3f} "
              f"{np.nanmean([r['tracking_rmse'] for r in R]):8.2f} "
              f"{np.max([r['max_lat_accel'] for r in R]):7.2f} "
              f"{np.percentile([r['max_jerk'] for r in R], 95):8.2f}")
    fails = [r for r in rows if not r["success"]]
    if fails:
        def why(r):
            if r["collided"]:
                return f"collision with {r['collision_with']!r} at {r['collision_speed']:.1f} m/s"
            return "off-road" if r["off_road"] else "stalled (no progress)"
        print("\nfailures:\n  " + "\n  ".join(f"{r['scenario']}#{r['seed']}: {why(r)}" for r in fails))
    print(f"\noverall pass rate: {np.mean([r['success'] for r in rows]) * 100:.1f}%  -> {a.out}")


if __name__ == "__main__":
    main()
