"""Does SAM keep each object's identity through the occlusion events of a CARLA sequence?

Matching per frame as in the MOTS20 occlusion study: ground-truth masks vs tracker masks, one-to-one (Hungarian) on
mask IoU >= 0.5. For every occlusion event in summary.json: SAM's ID in the last matched frame within 1 s before and
the first matched frame within 2 s after -> kept / switched / lost. Also: per-track coverage and ID changes overall.

    python sam_identity_check.py runs/ped_behind_van runs/ped_behind_van/sam/SAM3/data.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment

FPS, PRE_S, POST_S, IOU = 30, 1.0, 2.0, 0.5


def load(path):
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        f, tid, cls, h, w, rle = line.split(" ", 5)
        out[int(f)][int(tid)] = {"size": [int(h), int(w)], "counts": rle.strip().encode()}
    return out


seq, trk_path = Path(sys.argv[1]), Path(sys.argv[2])
gt, trk = load(seq / "label_02.txt"), load(trk_path)
ids = {}                                               # (gt track, frame) -> tracker id
for f, g in gt.items():
    t = trk.get(f, {})
    if not t:
        continue
    gk, tk = list(g), list(t)
    iou = np.asarray(mask_util.iou([t[k] for k in tk], [g[k] for k in gk], [0] * len(gk))).reshape(len(tk), len(gk)).T
    for r, c in zip(*linear_sum_assignment(-iou)):
        if iou[r, c] >= IOU:
            ids[(gk[r], f)] = tk[c]

summary = json.loads((seq / "summary.json").read_text())
events = []
for ev in summary["occlusion_events_seen_before_and_after"]:
    g, s, e = ev["track"], ev["start_frame"], ev["end_frame"]
    pre = next((ids[(g, f)] for f in range(s - 1, s - 1 - int(PRE_S * FPS), -1) if (g, f) in ids), None)
    post = next((ids[(g, f)] for f in range(e + 1, e + 1 + int(POST_S * FPS)) if (g, f) in ids), None)
    outcome = "not tracked before" if pre is None else ("lost" if post is None else ("kept" if pre == post else "switched"))
    events.append({**ev, "sam_id_before": pre, "sam_id_after": post, "outcome": outcome})

per_track = {}
for g in sorted({k for fr in gt.values() for k in fr}):
    frames = sorted(f for f, fr in gt.items() if g in fr)
    matched = [ids[(g, f)] for f in frames if (g, f) in ids]
    per_track[g] = {"gt_frames": len(frames), "matched_frames": len(matched),
                    "sam_ids_used": sorted(set(matched)), "id_changes": sum(1 for a, b in zip(matched, matched[1:]) if a != b)}
result = {"tracker": str(trk_path), "occlusion_events": events, "per_track": per_track}
print(json.dumps(result, indent=1))
(trk_path.parent / "identity_check.json").write_text(json.dumps(result, indent=1))
