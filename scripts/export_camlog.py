"""Drive a scenario closed-loop and write a camera log (depth / semantic / instance / RGB).

    PYTHONPATH=. .venv/bin/python scripts/export_camlog.py crosswalk_multiple_threat
    PYTHONPATH=. .venv/bin/python scripts/export_camlog.py occluded_pedestrian --cameras front front_tele \\
        --size 1280x704 --out logs/

The log feeds: SAM 3.1 auto-labelling (scripts/sam31_label.py, on a CUDA GPU),
perception scoring (scripts/score_perception.py) and Cosmos-Transfer (semantic.mp4 /
depth.mp4 as control videos, rgb.mp4 as the reference).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from avstack.config import StackConfig
from avstack.runner import run
from avstack.sim.camlog import CamLogWriter
from avstack.sim.scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--cameras", nargs="*", default=["front"])
    ap.add_argument("--size", default="960x540", help="render resolution WxH")
    ap.add_argument("--every", type=int, default=5, help="sim steps per frame (5 -> 10 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs")
    ap.add_argument("--no-videos", action="store_true")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    scenario, cfg = SCENARIOS[a.scenario](), StackConfig()
    out = Path(a.out) / f"{a.scenario}_seed{a.seed}"
    writer = CamLogWriter(out, scenario, cfg, a.cameras, w, h, a.every, seed=a.seed)
    t0 = time.time()
    result = run(scenario, cfg, seed=a.seed, on_step=writer.on_step)
    writer.close(videos=not a.no_videos)
    print(f"{'PASS' if result.success else 'FAIL'} {a.scenario}: {writer.n} frames x {len(a.cameras)} camera(s) "
          f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
