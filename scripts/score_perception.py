"""Score perception on a camera log against its ground truth (2D boxes, per camera).

    PYTHONPATH=. .venv/bin/python scripts/score_perception.py logs/crosswalk_multiple_threat_seed0 \\
        --camera front --pred tracks sam31

Predictors:
  tracks  the AV stack's own confirmed tracks (3D, world frame) projected into the camera
  sam31   <camera>/sam31_masklets.json from scripts/sam31_label.py (SAM 3.1 video masklets)
  sam3    <camera>/sam3_image_detections.json (per-frame SAM 3; detection metrics only)

Ground truth comes from the instance maps: every actor covering >= --min-pixels pixels in
the frame. Metrics (class-agnostic matching at IoU >= --iou, then class accuracy on matches):
  precision / recall, MOTA = 1 - (FN + FP + IDSW) / GT, IDF1 (global id correspondence),
  ID switches, and class accuracy (canonical class of the prediction vs the actor's class).
For real logs without ground truth, --reference sam31 scores the stack against SAM 3.1.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from avstack.common.types import ObjectClass
from avstack.perception.open_vocab.ontology import canonical_class


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])          # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


# ---------------------------------------------------------------- ground truth
def load_gt(log: Path, cam: str, min_pixels: int, amodal: bool, min_visibility: float = 0.25) -> dict[int, list]:
    """Ground truth per frame. ``amodal``: the whole projected 3D box (for 3D trackers);
    otherwise the visible-pixel box (for mask predictors like SAM). Objects that are too small
    or mostly hidden are kept as *don't care*: matching them is neither rewarded nor punished."""
    c = _camera(log, cam)
    frames = [json.loads(line) for line in (log / "frames.jsonl").read_text().splitlines()]
    gt = {}
    for fr in frames:
        inst = np.array(Image.open(log / cam / "instance" / f"{fr['i']:05d}.png"))
        ids, counts = np.unique(inst[inst > 0], return_counts=True)
        pixels = dict(zip((ids - 1).tolist(), counts.tolist()))
        items = []
        for o in fr["objects"]:
            n = pixels.get(o["id"], 0)
            if amodal:
                box = project_box(o, fr["ego"], c, o["height"])
                if box is None:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                vis = min(n / max(area, 1.0), 1.0)
            else:
                if n == 0:
                    continue
                ys, xs = np.nonzero(inst == o["id"] + 1)
                box, vis = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], 1.0
            items.append({"id": o["id"], "cls": o["cls"], "box": box,
                          "ignore": n < min_pixels or vis < min_visibility})
        gt[fr["i"]] = items
    return gt


# ---------------------------------------------------------------- predictions
def load_masklets(path: Path) -> dict[int, list]:
    d = json.loads(path.read_text())
    out = {}
    for k, items in d["frames"].items():
        out[int(k)] = [{"id": it["id"], "cls": canonical_class(it["prompt"]).value, "box": it["box"],
                        "score": it["score"]} for it in items]
    return out


def _camera(log: Path, cam: str) -> dict:
    meta = json.loads((log / "meta.json").read_text())
    return next(x for x in meta["cameras"] if x["name"] == cam)


def project_box(obj: dict, ego: dict, c: dict, height: float) -> list | None:
    """World 3D box (footprint x height) -> clipped 2D box in this camera, or None if not in view.
    Same calibration and mounting as the renderer."""
    m = c["mount"]
    psi = ego["yaw"] + m["yaw"]
    ox, oy = ego["x"] + m["mount_x"] * np.cos(ego["yaw"]), ego["y"] + m["mount_x"] * np.sin(ego["yaw"])
    ca, sa = np.cos(obj["yaw"]), np.sin(obj["yaw"])
    pts = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        wx = obj["x"] + sx * obj["length"] / 2 * ca - sy * obj["width"] / 2 * sa
        wy = obj["y"] + sx * obj["length"] / 2 * sa + sy * obj["width"] / 2 * ca
        fwd = (wx - ox) * np.cos(psi) + (wy - oy) * np.sin(psi)
        left = -(wx - ox) * np.sin(psi) + (wy - oy) * np.cos(psi)
        pts += [(fwd, left, 0.0), (fwd, left, height)]
    pts = np.array(pts)
    if (pts[:, 0] < 0.5).any():
        return None                                        # partly behind the camera
    u = c["cx"] - c["fx"] * pts[:, 1] / pts[:, 0]
    v = c["cy"] + c["fy"] * (m["mount_z"] - pts[:, 2]) / pts[:, 0]
    box = [max(u.min(), 0), max(v.min(), 0), min(u.max(), c["width"]), min(v.max(), c["height"])]
    return box if box[2] - box[0] >= 1 and box[3] - box[1] >= 1 else None


def project_tracks(log: Path, cam: str) -> dict[int, list]:
    """Stack tracks (world 3D boxes) -> amodal 2D boxes in this camera."""
    c = _camera(log, cam)
    out = {}
    for line in (log / "frames.jsonl").read_text().splitlines():
        fr = json.loads(line)
        items = []
        for t in fr["tracks"]:
            box = project_box(t, fr["ego"], c, ObjectClass(t["cls"]).typical_dims[2])
            if box is not None:
                items.append({"id": t["id"], "cls": t["cls"], "box": box, "score": 1.0})
        out[fr["i"]] = items
    return out


# ---------------------------------------------------------------- metrics
def evaluate(gt: dict, pred: dict, iou_thr: float) -> dict:
    tp = fp = fn = idsw = n_gt = cls_ok = 0
    last_match: dict[int, int] = {}
    pair_counts: dict[tuple, int] = defaultdict(int)
    gt_count: dict[int, int] = defaultdict(int)
    pr_count: dict[int, int] = defaultdict(int)
    per_class = defaultdict(lambda: [0, 0])                  # cls -> [matched, total]
    for f in sorted(gt):
        g, p = gt[f], pred.get(f, [])
        n_gt += sum(not x["ignore"] for x in g)
        for x in g:
            if not x["ignore"]:
                gt_count[x["id"]] += 1
                per_class[x["cls"]][1] += 1
        for y in p:
            pr_count[y["id"]] += 1
        iou = iou_matrix(np.array([x["box"] for x in g], float).reshape(-1, 4),
                         np.array([y["box"] for y in p], float).reshape(-1, 4))
        matched = []
        if iou.size:
            r, cidx = linear_sum_assignment(-iou)
            matched = [(i, j) for i, j in zip(r, cidx) if iou[i, j] >= iou_thr]
        # don't-care: a prediction matched to an ignored object is neither TP nor FP; an
        # ignored object left unmatched is not a miss
        ign_match = [(i, j) for i, j in matched if g[i]["ignore"]]
        matched = [(i, j) for i, j in matched if not g[i]["ignore"]]
        n_ign = sum(x["ignore"] for x in g)
        # unmatched predictions overlapping an ignored object are forgiven too
        used = {j for _, j in matched} | {j for _, j in ign_match}
        ign_idx = [i for i, x in enumerate(g) if x["ignore"]]
        forgiven = {j for j in range(len(p)) if j not in used and ign_idx and iou[ign_idx, j].max() >= 0.3}
        n_gt += 0
        tp += len(matched)
        fp += len(p) - len(matched) - len(ign_match) - len(forgiven)
        fn += len(g) - n_ign - len(matched)
        for i, j in matched:
            gid, pid = g[i]["id"], p[j]["id"]
            per_class[g[i]["cls"]][0] += 1
            same_family = ObjectClass(g[i]["cls"]).family == ObjectClass(p[j]["cls"]).family
            cls_ok += (g[i]["cls"] == p[j]["cls"]) or (same_family and g[i]["cls"] in ("car", "truck", "bus"))
            if pid >= 0:
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                last_match[gid] = pid
                pair_counts[(gid, pid)] += 1
    # IDF1: best one-to-one assignment between gt ids and predicted ids over the whole log
    gids, pids = sorted(gt_count), sorted(i for i in pr_count if i >= 0)
    idtp = 0
    if gids and pids and pair_counts:
        M = np.zeros((len(gids), len(pids)))
        gi, pi = {g: k for k, g in enumerate(gids)}, {p: k for k, p in enumerate(pids)}
        for (g, p), n in pair_counts.items():
            M[gi[g], pi[p]] = n
        r, cidx = linear_sum_assignment(-M)
        idtp = int(M[r, cidx].sum())
    n_pred = sum(pr_count[i] for i in pids)
    return {
        "frames": len(gt), "gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1),
        "mota": 1 - (fn + fp + idsw) / max(n_gt, 1),
        "idf1": 2 * idtp / max(n_gt + n_pred, 1) if pids else float("nan"),
        "class_acc": cls_ok / max(tp, 1),
        "recall_by_class": {k: v[0] / max(v[1], 1) for k, v in sorted(per_class.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--pred", nargs="+", default=["tracks"], choices=["tracks", "sam31", "sam3"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-pixels", type=int, default=150)
    a = ap.parse_args()
    gt_amodal = load_gt(a.log, a.camera, a.min_pixels, amodal=True)
    gt_modal = load_gt(a.log, a.camera, a.min_pixels, amodal=False)
    sources = {"tracks": lambda: project_tracks(a.log, a.camera),
               "sam31": lambda: load_masklets(a.log / a.camera / "sam31_masklets.json"),
               "sam3": lambda: load_masklets(a.log / a.camera / "sam3_image_detections.json")}
    n_care = sum(not x["ignore"] for v in gt_amodal.values() for x in v)
    print(f"{a.log.name} / {a.camera}: {n_care} scored GT objects over {len(gt_amodal)} frames "
          f"(>= {a.min_pixels} px visible, >= 25% visible), IoU >= {a.iou}")
    for name in a.pred:
        pred = sources[name]()
        gt = gt_amodal if name == "tracks" else gt_modal     # 3D tracks are amodal, masks are modal
        if name == "sam3":                      # per-frame model: score only frames it covered
            gt_eval = {f: gt[f] for f in pred if f in gt}
        else:
            gt_eval = gt
        m = evaluate(gt_eval, pred, a.iou)
        print(f"\n[{name}] frames={m['frames']}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"MOTA={m['mota']:.3f}  IDF1={m['idf1']:.3f}  IDsw={m['idsw']}  class_acc={m['class_acc']:.3f}")
        print("  recall by class: " + ", ".join(f"{k}={v:.2f}" for k, v in m["recall_by_class"].items()))


if __name__ == "__main__":
    main()
