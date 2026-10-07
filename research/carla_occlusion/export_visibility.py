"""Ground truth for every car and pedestrian: masks (KITTI-MOTS format), exact visibility, and the review video.

Instance IDs in CARLA 0.9.14 are not actor IDs, so each actor is linked to its instance ID geometrically: its 3D box is
projected into the camera and the instance of the right class covering most of the box wins, by majority over the
sequence (an object's instance ID never changes). Actors are linked across the two passes by type and start position.
visible mask = the object's pixels in the full pass; full outline = its pixels in the amodal pass (occluders removed);
visibility = |visible| / |outline| (parked occluders are absent in the amodal pass: visibility not defined for them).

    python export_visibility.py runs/ped_behind_van
"""
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

VEH_TAGS, PED_TAGS = {14, 15, 16, 18, 19}, {12}       # CARLA 0.9.14 (cityscapes ids): car, truck, bus, motorcycle, bicycle
OCC_VIS = 0.25
root = Path(sys.argv[1])
P = {p: json.loads((root / p / "meta.json").read_text()) for p in ("full", "amodal")}
n = min(len(P["full"]["frames"]), len(P["amodal"]["frames"]))
K = np.array(P["full"]["calib"]["K"]); W, H = P["full"]["calib"]["image_size"]


def load_inst(p, i):
    im = cv2.imread(str(root / p / "inst" / f"{i:06d}.png")).astype(np.int32)     # B, G, R of the sensor's BGRA
    return im[:, :, 2], im[:, :, 1] + 256 * im[:, :, 0]                           # semantic tag, instance id


def project_box(o, cam_inv):
    ex, loc = np.array(o["bb_ext"]), np.array(o["bb_loc"])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ex + loc
    world = np.array(o["M"]) @ np.c_[corners, np.ones(8)].T
    cam = np.array(cam_inv) @ world                          # UE camera frame: x forward, y right, z up
    x, y, z = cam[0], cam[1], cam[2]
    if (x < 0.5).all():
        return None
    x = np.maximum(x, 0.5)
    u, v = K[0, 2] + K[0, 0] * y / x, K[1, 2] - K[1, 1] * z / x
    u0, u1, v0, v1 = int(max(0, u.min())), int(min(W, u.max())), int(max(0, v.min())), int(min(H, v.max()))
    return (u0, v0, u1, v1) if u1 > u0 and v1 > v0 else None


# 1) link actors to instance ids: one-to-one per frame (Hungarian on pixel counts inside each projected box),
#    then one-to-one over the summed votes of the sequence
from scipy.optimize import linear_sum_assignment  # noqa: E402

inst_of = {}
for p in P:
    votes = defaultdict(Counter)
    for i in range(0, n, 3):
        tag, iid = load_inst(p, i)
        fr = P[p]["frames"][i]
        cand = {}
        for o in fr["objects"]:
            box = project_box(o, fr["cam_inv"])
            if box is None:
                continue
            u0, v0, u1, v1 = box
            t, d = tag[v0:v1, u0:u1], iid[v0:v1, u0:u1]
            ok = np.isin(t, list(PED_TAGS if o["cls"] == 2 else VEH_TAGS))
            if ok.sum() >= 15:
                ids, c = np.unique(d[ok], return_counts=True)
                cand[o["id"]] = dict(zip(ids.tolist(), c.tolist()))
        if not cand:
            continue
        aids = list(cand)
        iids = sorted({x for v in cand.values() for x in v})
        S = np.array([[cand[a].get(x, 0) for x in iids] for a in aids], float)
        for r, c in zip(*linear_sum_assignment(-S)):
            if S[r, c] >= 15:
                votes[aids[r]][iids[c]] += 1
    aids = list(votes)
    iids = sorted({x for v in votes.values() for x in v})
    V = np.array([[votes[a].get(x, 0) for x in iids] for a in aids], float)
    inst_of[p] = {aids[r]: iids[c] for r, c in zip(*linear_sum_assignment(-V)) if V[r, c] > 0}

# 2) stable track ids; link the amodal actor to the full-pass actor by type and first-frame position
f0, a0 = P["full"]["frames"][0]["objects"], P["amodal"]["frames"][0]["objects"]
link = {}
for o in f0:
    cand = [x for x in a0 if x["type"] == o["type"] and np.hypot(x["M"][0][3] - o["M"][0][3], x["M"][1][3] - o["M"][1][3]) < 0.5]
    if cand:
        link[o["id"]] = cand[0]["id"]
tracks, cnt = {}, {1: 0, 2: 0}
for o in sorted(f0, key=lambda x: (x["cls"], x["id"])):
    if o["id"] in inst_of["full"]:
        cnt[o["cls"]] += 1
        tracks[o["id"]] = {"track": o["cls"] * 1000 + cnt[o["cls"]], "cls": o["cls"], "type": o["type"],
                           "amodal": link.get(o["id"])}

# 3) per frame: masks, visibility, GT lines, video
rows, gt, ious = [], [], []
cap = cv2.VideoCapture(str(root / "full" / "chase_hud.mp4"))
OW = 1280; KH2 = round(H * OW / W) // 2 * 2
vw = cv2.VideoWriter(str(root / "_tmp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (OW, 720 + KH2))


PALETTE = [(230, 120, 40), (200, 60, 200), (60, 200, 60), (230, 200, 40), (60, 60, 230), (200, 200, 200)]   # BGR, no orange


def palette(tid):
    return PALETTE[(tid % 1000 + (0 if tid < 2000 else 3)) % len(PALETTE)]


for i in range(n):
    _, iid_f = load_inst("full", i)
    _, iid_a = load_inst("amodal", i)
    rgb = cv2.imread(str(root / "full" / "rgb" / f"{i:06d}.png"))
    over, labels = rgb.copy(), []
    for aid, tr in tracks.items():
        vis = iid_f == inst_of["full"][aid]
        v_px = int(vis.sum())
        amo = (iid_a == inst_of["amodal"][tr["amodal"]]) if tr["amodal"] in inst_of["amodal"] else None
        a_px = int(amo.sum()) if amo is not None else None
        visibility = (v_px / a_px) if a_px else None
        if a_px and a_px > 40 and v_px / a_px > 0.9:
            ious.append((vis & amo).sum() / (vis | amo).sum())
        rows.append({"frame": i, "track": tr["track"], "class": "pedestrian" if tr["cls"] == 2 else "car", "type": tr["type"],
                     "visible_px": v_px, "full_px": a_px if a_px is not None else "",
                     "visibility": round(visibility, 4) if visibility is not None else ""})
        if v_px:
            rle = mask_util.encode(np.asfortranarray(vis.astype(np.uint8)))
            gt.append(f"{i} {tr['track']} {tr['cls']} {H} {W} {rle['counts'].decode()}")
            col = palette(tr["track"])
            over[vis] = (0.5 * over[vis] + 0.5 * np.array(col)).astype(np.uint8)
            ys, xs = np.nonzero(vis)
            labels.append((f"#{tr['track']}" + (f" {min(visibility, 1.0):.0%}" if visibility is not None else ""), int(xs.min()), int(ys.min()), col))
        if amo is not None and a_px:
            hidden = amo & ~vis
            over[hidden] = (0.5 * over[hidden] + 0.5 * np.array([0, 165, 255])).astype(np.uint8)
            cnts, _ = cv2.findContours(amo.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, (0, 165, 255), 1, cv2.LINE_AA)
            if not v_px:
                ys, xs = np.nonzero(amo)
                labels.append((f"#{tr['track']} hidden", int(xs.min()), int(ys.min()), (0, 165, 255)))
    ok, top = cap.read()
    if not ok:
        break
    bot = cv2.resize(over, (OW, KH2), interpolation=cv2.INTER_AREA)
    sx, sy = OW / W, KH2 / H
    for text, x, y, col in labels:
        x, y = int(x * sx), max(44, int(y * sy) - 4)
        cv2.putText(bot, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bot, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.rectangle(bot, (0, 0), (OW, 30), (20, 20, 20), -1)
    cv2.putText(bot, "KITTI front camera (SAM input): ground truth for every car and pedestrian, #track visibility; "
                     "orange = hidden part of the full outline", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    vw.write(np.vstack([top, bot]))
vw.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(root / "_tmp.mp4"), "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(root / "pedestrian_behind_van.mp4")], check=True)
(root / "_tmp.mp4").unlink()
with open(root / "visibility.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(root / "label_02.txt").write_text("\n".join(gt) + "\n")

# occlusion events per track: visibility < 0.25, visible (>= 0.25) just before and just after
events = []
for tr in tracks.values():
    seq = [r["visibility"] if r["visibility"] != "" else None for r in rows if r["track"] == tr["track"]]
    k = 0
    while k < len(seq):
        if seq[k] is not None and seq[k] < OCC_VIS:
            j = k
            while j < len(seq) and seq[j] is not None and seq[j] < OCC_VIS:
                j += 1
            before = k > 0 and seq[k - 1] is not None and seq[k - 1] >= OCC_VIS
            after = j < len(seq) and seq[j] is not None and seq[j] >= OCC_VIS
            if before and after:
                events.append({"track": tr["track"], "start_frame": k, "end_frame": j - 1,
                               "seconds": round((j - k) * P["full"]["dt"], 2), "min_visibility": round(min(seq[k:j]), 3)})
            k = j
        else:
            k += 1
summary = {"frames": n, "tracks": {tr["track"]: tr["type"] for tr in tracks.values()},
           "mask_agreement_unoccluded_iou": round(float(np.mean(ious)), 3) if ious else None,
           "occlusion_events_seen_before_and_after": events, "collisions": P["full"].get("collisions"),
           "gt_lines": len(gt)}
(root / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
