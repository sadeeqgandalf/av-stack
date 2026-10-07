"""Run SAM 3 / SAM 3.1 on a CARLA sequence exactly as in the KITTI-MOTS / MOTS20 benchmark.

Reuses the benchmark runner (research/sam_mots_benchmark/run_sam_mots.py): same builders and settings, same official
flow (text prompt on frame 0, propagate), same MOTS conversion. Prompts: "car" (class 1), "pedestrian" (class 2).

    python run_sam_carla.py --model sam3 --seq runs/ped_behind_van --out runs/ped_behind_van/sam
"""
import argparse
import json
import time
from pathlib import Path

import run_sam_mots as bench                      # benchmark runner, copied next to this file

CLASSES = {"car": 1, "pedestrian": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sam3", "sam3.1"], required=True)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    frames_dir = a.seq / "full" / "rgb"
    frames = sorted(frames_dir.glob("*.png"))
    h, w = 375, 1242
    pred, build_info = bench.build(a.model)
    t0 = time.time()
    sess = pred.handle_request({"type": "start_session", "resource_path": str(frames_dir), **bench.SESSION_OPTIONS})["session_id"]
    per_class = {cid: bench.run_prompt(pred, sess, text) for text, cid in CLASSES.items()}
    pred.handle_request({"type": "close_session", "session_id": sess})
    tracker = "SAM3" if a.model == "sam3" else "SAM3.1"
    out = a.out / tracker
    out.mkdir(parents=True, exist_ok=True)
    stats = bench.write_sequence(out / "data.txt", per_class, len(frames), 0, h, w)
    stats.update(frames=len(frames), seconds=round(time.time() - t0, 1), build=build_info)
    (out / "run_info.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "build"}, default=str))


if __name__ == "__main__":
    main()
