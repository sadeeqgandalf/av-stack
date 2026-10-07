#!/usr/bin/env python3
"""Identity through occlusion on MOTS20 train (see OCCLUSION_PROTOCOL.md; thresholds fixed there).

    python occlusion_analysis.py --mots-gt TrackEval/data/gt/mot_challenge/MOTS-train \
        --mot17 occlusion/mot17/train --trackers trackers/mots_challenge --out occlusion/results
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

SEQS = ["MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"]
TRACKERS = ["SAM3", "SAM3.1", "track_rcnn"]      # track_rcnn: TrackR-CNN reference (TrackEval data.zip)
LINK_VIS, LINK_IOU = 0.5, 0.5          # MOTS20 <-> MOT17 linking
OCC_VIS = 0.25                         # occluded: visibility < 0.25
HEAVY_VIS = 0.1                        # depth: min visibility < 0.1 = heavy/full
MATCH_IOU = 0.5                        # tracker <-> GT mask match
PRE_S, POST_S = 1.0, 2.0               # search windows before / after an event
CTRL_VIS, CTRL_S = 0.5, 1.0            # control: stays visible >= 0.5 for 1 s
DUR_BINS = [(0, 0.5, "<0.5 s"), (0.5, 1, "0.5-1 s"), (1, 2, "1-2 s"), (2, 1e9, ">=2 s")]
N_BOOT, SEED = 10_000, 0


def load_mots(path: Path, keep_class=2) -> dict[int, dict[int, dict]]:
    """{frame: {id: rle}} for one class (1-based frames)."""
    out = defaultdict(dict)
    for line in path.read_text().splitlines():
        f, tid, cls, h, w, rle = line.split(" ", 5)
        if int(cls) == keep_class:
            out[int(f)][int(tid)] = {"size": [int(h), int(w)], "counts": rle.strip().encode()}
    return out


def load_mot17(seq_dir: Path):
    """{track: {frame: (box xywh, visibility)}} for pedestrians with consider-flag 1, and fps."""
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    fps = float(ini["Sequence"]["frameRate"])
    tracks = defaultdict(dict)
    for row in csv.reader(open(seq_dir / "gt" / "gt.txt")):
        f, tid, x, y, w, h, flag, cls, vis = row[:9]
        if int(cls) == 1 and int(flag) == 1:
            tracks[int(tid)][int(f)] = (np.array([float(x), float(y), float(w), float(h)]), float(vis))
    return tracks, fps


def box_iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def hungarian(iou: np.ndarray, thr: float):
    if iou.size == 0:
        return []
    r, c = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]


def link_people(gt, mot17):
    """MOTS20 person id -> MOT17 track id (majority vote over per-frame box matches)."""
    votes = defaultdict(Counter)
    for f, people in gt.items():
        cand = [(t, fr[f][0]) for t, fr in mot17.items() if f in fr and fr[f][1] >= LINK_VIS]
        if not cand or not people:
            continue
        pids = list(people)
        boxes = mask_util.toBbox([people[p] for p in pids])
        iou = np.array([[box_iou(b, cb) for _, cb in cand] for b in boxes])
        for i, j in hungarian(iou, LINK_IOU):
            votes[pids[i]][cand[j][0]] += 1
    link, agree, total = {}, 0, 0
    for pid, c in votes.items():
        t, n = c.most_common(1)[0]
        link[pid] = t
        agree += n
        total += sum(c.values())
    return link, agree / total if total else float("nan"), len(set(p for fr in gt.values() for p in fr) - set(link))


def tracker_ids(gt, trk):
    """{(person, frame): tracker id} from per-frame one-to-one mask-IoU matching."""
    out = {}
    for f, people in gt.items():
        dets = trk.get(f, {})
        if not dets:
            continue
        pids, tids = list(people), list(dets)
        iou = mask_util.iou([dets[t] for t in tids], [people[p] for p in pids], [0] * len(pids)).T
        for i, j in hungarian(np.asarray(iou).reshape(len(pids), len(tids)), MATCH_IOU):
            out[(pids[i], f)] = tids[j]
    return out


def find_events(track: dict, fps: float):
    frames = sorted(track)
    vis = {f: track[f][1] for f in frames}
    events, f_i = [], 0
    while f_i < len(frames):
        f = frames[f_i]
        if vis[f] < OCC_VIS:
            start = f_i
            while f_i < len(frames) and vis[frames[f_i]] < OCC_VIS and (f_i == start or frames[f_i] == frames[f_i - 1] + 1):
                f_i += 1
            s, e = frames[start], frames[f_i - 1]
            before, after = s - 1, e + 1
            if vis.get(before, -1) >= OCC_VIS and vis.get(after, -1) >= OCC_VIS:
                n = e - s + 1
                events.append({"start": s, "end": e, "frames": n, "seconds": n / fps,
                               "min_vis": min(vis[x] for x in range(s, e + 1))})
        else:
            f_i += 1
    return events


def outcome(ids: dict, pid: int, ev: dict, fps: float):
    pre = next((ids[(pid, f)] for f in range(ev["start"] - 1, ev["start"] - 1 - int(round(PRE_S * fps)), -1)
                if (pid, f) in ids), None)
    if pre is None:
        return "not_tracked_before"
    post = next((ids[(pid, f)] for f in range(ev["end"] + 1, ev["end"] + 1 + int(round(POST_S * fps)))
                 if (pid, f) in ids), None)
    if post is None:
        return "lost"
    return "kept" if post == pre else "switched"


def control(ids, pid, mframes, track, fps):
    """Non-overlapping 1-s windows in which the person stays visible >= CTRL_VIS: same ID after 1 s?"""
    d = int(round(CTRL_S * fps))
    res, f = [], min(mframes) if mframes else 0
    last = max(mframes) if mframes else -1
    while f + d <= last:
        window_ok = all(track.get(x, (None, -1))[1] >= CTRL_VIS for x in range(f, f + d + 1))
        if window_ok and (pid, f) in ids:
            a, b = ids[(pid, f)], ids.get((pid, f + d))
            res.append("lost" if b is None else ("kept" if a == b else "switched"))
            f += d
        else:
            f += 1
    return res


def rate(outs, num=("kept",), den=("kept", "switched", "lost")):
    n = sum(o in den for o in outs)
    return (sum(o in num for o in outs) / n) if n else float("nan"), n


def boot_ci(groups: dict, num, den, rng):
    """Percentile bootstrap over people: groups = {person_key: [outcome, ...]}."""
    keys = list(groups)
    if not keys:
        return (float("nan"), float("nan"))
    vals = []
    arr = [np.array([o in num for o in groups[k]]) for k in keys]
    dn = [np.array([o in den for o in groups[k]]) for k in keys]
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        n = sum(dn[i].sum() for i in idx)
        vals.append(sum((arr[i] & dn[i]).sum() for i in idx) / n if n else np.nan)
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mots-gt", type=Path, required=True)
    ap.add_argument("--mot17", type=Path, required=True)
    ap.add_argument("--trackers", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="TrackR-CNN tracker folder (contains data/)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows, ctrl_rows, link_report = [], [], {}
    for seq in SEQS:
        gt = load_mots(a.mots_gt / seq / "gt" / "gt.txt")
        mot17, fps = load_mot17(a.mot17 / f"MOT17-{seq[-2:]}-FRCNN")
        link, agree, unlinked = link_people(gt, mot17)
        link_report[seq] = {"people": len({p for fr in gt.values() for p in fr}), "linked": len(link),
                            "unlinked": unlinked, "vote_agreement": round(agree, 4), "fps": fps}
        person_frames = defaultdict(list)
        for f, people in gt.items():
            for p in people:
                person_frames[p].append(f)
        tdir = {t: (a.reference if t == "track_rcnn" else a.trackers / t) for t in TRACKERS}
        ids = {t: tracker_ids(gt, load_mots(tdir[t] / "data" / f"{seq}.txt")) for t in TRACKERS}
        for pid, tid17 in link.items():
            track = mot17[tid17]
            for ev in find_events(track, fps):
                row = {"seq": seq, "person": pid, "mot17_id": tid17, **ev}
                for t in TRACKERS:
                    row[t] = outcome(ids[t], pid, ev, fps)
                rows.append(row)
            for t in TRACKERS:
                for o in control(ids[t], pid, sorted(person_frames[pid]), track, fps):
                    ctrl_rows.append({"seq": seq, "person": pid, "tracker": t, "outcome": o})
        print(seq, link_report[seq], "events:", sum(r["seq"] == seq for r in rows), flush=True)

    with open(a.out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def rng_for(label, t):                       # independent stream per (subset, tracker): CIs do not depend on the tracker set
        return np.random.default_rng([SEED, zlib.crc32(f"{label}|{t}".encode())])
    K, KSL = ("kept",), ("kept", "switched", "lost")

    def summarize(sel, label):
        out = {"subset": label}
        for t in TRACKERS:
            groups = defaultdict(list)
            for r in sel:
                groups[(r["seq"], r["person"])].append(r[t])
            outs = [r[t] for r in sel]
            c = Counter(outs)
            rec, n = rate(outs)
            reas, n2 = rate(outs, K, ("kept", "switched"))
            lo, hi = boot_ci(groups, K, KSL, rng_for(label, t))
            out.update({f"{t} events": n, f"{t} kept": c["kept"], f"{t} switched": c["switched"],
                        f"{t} lost": c["lost"], f"{t} not_tracked_before": c["not_tracked_before"],
                        f"{t} recovery": rec, f"{t} recovery_ci": [lo, hi], f"{t} reassociation": reas})
        return out

    subsets = [(rows, "all events")]
    for lo, hi, lab in DUR_BINS:
        subsets.append(([r for r in rows if lo <= r["seconds"] < hi], f"duration {lab}"))
    subsets.append(([r for r in rows if r["min_vis"] >= HEAVY_VIS], "depth partial (min vis 0.10-0.25)"))
    subsets.append(([r for r in rows if r["min_vis"] < HEAVY_VIS], "depth heavy/full (min vis < 0.10)"))
    for seq in SEQS:
        subsets.append(([r for r in rows if r["seq"] == seq], seq))
    summary = [summarize(s, lab) for s, lab in subsets]

    both = [r for r in rows if r["SAM3"] != "not_tracked_before" and r["SAM3.1"] != "not_tracked_before"]
    only3 = sum(r["SAM3"] == "kept" and r["SAM3.1"] != "kept" for r in both)
    only31 = sum(r["SAM3.1"] == "kept" and r["SAM3"] != "kept" for r in both)
    paired = {"events_tracked_before_by_both": len(both), "only_SAM3_kept": only3, "only_SAM3.1_kept": only31,
              "both_kept": sum(r["SAM3"] == "kept" and r["SAM3.1"] == "kept" for r in both),
              "neither_kept": sum(r["SAM3"] != "kept" and r["SAM3.1"] != "kept" for r in both),
              "mcnemar_exact_p": binomtest(only3, only3 + only31, 0.5).pvalue if only3 + only31 else float("nan")}

    ctrl = {}
    for t in TRACKERS:
        outs = [r["outcome"] for r in ctrl_rows if r["tracker"] == t]
        groups = defaultdict(list)
        for r in ctrl_rows:
            if r["tracker"] == t:
                groups[(r["seq"], r["person"])].append(r["outcome"])
        rec, n = rate(outs)
        ctrl[t] = {"windows": n, "same_id_rate": rec, "ci": boot_ci(groups, K, KSL, rng_for("control", t)), **Counter(outs)}

    result = {"linking": link_report, "summary": summary, "paired": paired, "control_no_occlusion": ctrl}
    (a.out / "occlusion_results.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"paired": paired, "control": ctrl}, indent=1, default=float))
    for s in summary:
        print(f"{s['subset']:<38}" + "  ".join(
            f"{t}: {s[f'{t} recovery']*100:5.1f}% [{s[f'{t} recovery_ci'][0]*100:4.1f}-{s[f'{t} recovery_ci'][1]*100:4.1f}] n={s[f'{t} events']}"
            for t in TRACKERS))


if __name__ == "__main__":
    main()
