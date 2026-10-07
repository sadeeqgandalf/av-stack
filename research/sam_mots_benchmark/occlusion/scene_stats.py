# Per-sequence scene statistics for MOTS20 train: crowding (MOTS20 GT) and occlusion (MOT17 visibility).
import json, sys
from collections import defaultdict
sys.path.insert(0, "/workspace/sam_mots_benchmark/occlusion")
import occlusion_analysis as oa
from pathlib import Path
root = Path("/workspace/sam_mots_benchmark")
out = {}
for seq in oa.SEQS:
    gt = oa.load_mots(root / f"TrackEval/data/gt/mot_challenge/MOTS-train/{seq}/gt/gt.txt")
    mot17, fps = oa.load_mot17(root / f"occlusion/mot17/train/MOT17-{seq[-2:]}-FRCNN")
    n_frames = max(max(t) for t in mot17.values())
    vis = [v for t in mot17.values() for (_, v) in t.values()]
    per_frame = defaultdict(int)
    for t in mot17.values():
        for f in t:
            per_frame[f] += 1
    h, w = next(iter(next(iter(gt.values())).values()))["size"]
    out[seq] = {"fps": fps, "frames": n_frames, "resolution": f"{w}x{h}",
                "gt_people": len({p for fr in gt.values() for p in fr}),
                "visible_people_per_frame": round(sum(len(fr) for fr in gt.values()) / n_frames, 1),
                "people_per_frame_incl_hidden": round(sum(per_frame.values()) / n_frames, 1),
                "pct_person_frames_occluded": round(100 * sum(v < oa.OCC_VIS for v in vis) / len(vis), 1)}
    print(seq, out[seq])
Path(root / "occlusion/results/scene_stats.json").write_text(json.dumps(out, indent=1))
