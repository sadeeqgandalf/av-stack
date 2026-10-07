"""Auto-label camera logs with SAM 3.1 (Object Multiplex): masklets with persistent ids.

Runs where SAM lives. On a CUDA GPU (your pod):

    export AVSTACK_SAM3_PYTHON=/path/to/.venv_sam3/bin/python
    PYTHONPATH=. python scripts/sam31_label.py logs/crosswalk_multiple_threat_seed0 --cameras front front_tele

Writes <log>/<camera>/sam31_masklets.json (per frame: id, prompt, score, box, area).
Prompts default to the log's vocabulary (meta.json); override with --prompts.

--mode image runs the per-frame SAM 3 image model instead (works on a Mac, slowly; ids are
per frame, so only detection metrics are meaningful). Use --max-frames for a quick check.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from avstack.perception.open_vocab.sam3_client import Sam3Service


def label_image_mode(svc: Sam3Service, cam_dir: Path, prompts, thr: float, max_frames: int | None) -> dict:
    frames = sorted((cam_dir / "rgb").glob("*.jpg"))[: max_frames or None]
    out = {"version": "sam3-image", "prompts": prompts, "threshold": thr, "frames": {}}
    for f in frames:
        r = svc.segment(str(f), prompts, thr)
        out["frames"][str(int(f.stem))] = [
            {"id": -1, "prompt": i["prompt"], "score": i["score"], "box": i["box"], "area": i["area"]}
            for i in r["instances"]]
        print(f"  {cam_dir.name} frame {f.stem}: {len(r['instances'])} instances ({r['ms'] / 1000:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--cameras", nargs="*", default=["front"])
    ap.add_argument("--prompts", nargs="*")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--mode", choices=["video", "image"], default="video")
    ap.add_argument("--version", default="sam3.1", choices=["sam3.1", "sam3"])
    ap.add_argument("--max-objects", type=int, default=32)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--sam3-python", help="overrides AVSTACK_SAM3_PYTHON")
    a = ap.parse_args()

    with Sam3Service(python=a.sam3_python, timeout=6 * 3600) as svc:
        print(f"SAM service on {svc.device}")
        for log in a.logs:
            meta = json.loads((log / "meta.json").read_text())
            prompts = a.prompts or meta["vocabulary"]
            for cam in a.cameras:
                cam_dir = log / cam
                out_json = cam_dir / ("sam31_masklets.json" if a.mode == "video" else "sam3_image_detections.json")
                t0 = time.time()
                if a.mode == "video":
                    rgb = cam_dir / "rgb"
                    if a.max_frames:                       # SAM reads a whole folder: stage a subset
                        rgb = cam_dir / f"_rgb_first{a.max_frames}"
                        rgb.mkdir(exist_ok=True)
                        for f in sorted((cam_dir / "rgb").glob("*.jpg"))[: a.max_frames]:
                            (rgb / f.name).write_bytes(f.read_bytes())
                    r = svc.track_video(str(rgb), prompts, str(out_json), a.threshold, a.version, a.max_objects)
                    print(f"{log.name}/{cam}: {r['frames']} frames, {r['objects']} objects "
                          f"({time.time() - t0:.0f}s) -> {out_json}")
                else:
                    out = label_image_mode(svc, cam_dir, prompts, a.threshold, a.max_frames)
                    out_json.write_text(json.dumps(out))
                    print(f"{log.name}/{cam}: {len(out['frames'])} frames ({time.time() - t0:.0f}s) -> {out_json}")


if __name__ == "__main__":
    main()
