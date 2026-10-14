#!/usr/bin/env python3
"""One occlusion event on MOTS20, side by side: SAM alone (left) and SAM + the EKF layer (right).

    python ekf_event_video.py <tracker_in> <seq> <person> <start> <end> <image_dir> <out.mp4>

<person> is the MOTS20 ground-truth id; <start> and <end> are the frames of the hide (from occlusion/events.csv).
The ground truth is used only to know which mask is this person, so that both panels follow the same one.
"""
import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pycocotools.mask as mask_util

import ekf_relink as E

sys.path.insert(0, str(Path(__file__).parent.parent / "occlusion"))
from occlusion_analysis import load_mots, tracker_ids  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD_S = 1.5                                   # seconds shown before and after the hide


def text(img, s, xy, scale=0.6, col=(255, 255, 255), thick=1):
    cv2.putText(img, s, xy, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, col, thick, cv2.LINE_AA)


def colour(i):
    c = np.random.default_rng(i * 7919).integers(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def main(src, seq, person, start, end, image_dir, out):
    person, start, end = int(person), int(start), int(end)
    src = Path(src) / "data" / f"{seq}.txt"
    gt = load_mots(E.DATA / f"x_data/data/gt/mot_challenge/MOTS-train/{seq}/gt/gt.txt")
    sam = E.read(src)
    h, w = next(iter(sam.values()))[0][2:4]
    cam = E.camera("mots", seq, h, w)
    layer, ekf, state = E.Layer(cam), {}, {}
    first, pad = min(sam), int(round(PAD_S / cam.dt))
    for k in range(first, min(end + pad, max(sam)) + 1):
        rows = sam.get(k, [])
        ids = layer.step(k - first, E.boxes(rows))
        ekf[k] = [(i, r[1], r[2], r[3], r[4]) for i, r in zip(ids, rows)]
        state[k] = {t["id"]: (t["x"][:2].copy(), t["P"][:2, :2].copy(), t["lost"]) for t in layer.tracks}
    who = {"SAM 3.1 alone": tracker_ids(gt, load_mots(src)), "SAM 3.1 + EKF": tracker_ids(gt, {k: {r[0]: {"size": [r[2], r[3]], "counts": r[4].encode()} for r in v} for k, v in ekf.items()})}
    rows_of = {"SAM 3.1 alone": sam, "SAM 3.1 + EKF": ekf}
    frames = range(max(start - pad, first), min(end + pad, max(sam)) + 1)

    bb = np.array([mask_util.toBbox(gt[f][person]) for f in frames if person in gt.get(f, {})])
    x0, y0 = bb[:, 0].min(), bb[:, 1].min(); x1, y1 = (bb[:, 0] + bb[:, 2]).max(), (bb[:, 1] + bb[:, 3]).max()
    side = min(int(max((x1 - x0) * 1.5, (y1 - y0) * 1.5, 420)), h, w)      # at least 420 px, so the scene around is visible
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(np.clip(cx - side / 2, 0, max(w - side, 0))), int(np.clip(cy - side / 2, 0, max(h - side, 0)))
    S = 640
    before = {name: next((who[name][(person, f)] for f in range(start - 1, start - 1 - pad, -1) if (person, f) in who[name]), None) for name in who}

    writer = imageio.get_writer(out, fps=round(1 / cam.dt) / 3, codec="libx264", quality=8, macro_block_size=1)   # one-third speed
    for f in frames:
        img = cv2.imread(str(Path(image_dir) / f"{f:06d}.jpg"))
        panels = []
        for name in who:
            pic, tid, ghost = img.copy(), who[name].get((person, f)), False
            for r in rows_of[name].get(f, []):
                m = mask_util.decode({"size": [r[2], r[3]], "counts": r[4].encode()}).astype(bool)
                if r[0] == tid:
                    pic[m] = (0.45 * pic[m] + 0.55 * np.array(colour(r[0]))).astype(np.uint8)
                    cv2.drawContours(pic, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 2)
            if name.endswith("EKF") and before[name] in state[f] and state[f][before[name]][2] > 0:
                x, P, lost = state[f][before[name]]              # out of sight: the filter keeps predicting
                ghost = True
                val, vec = np.linalg.eigh(P)
                cv2.ellipse(pic, (int(x[0]), int(x[1])), (int(2 * math.sqrt(val[1])), int(2 * math.sqrt(val[0]))),
                            math.degrees(math.atan2(vec[1, 1], vec[0, 1])), 0, 360, (80, 255, 80), 2, cv2.LINE_AA)
                cv2.drawMarker(pic, (int(x[0]), int(x[1])), (80, 255, 80), cv2.MARKER_CROSS, 18, 2)
            pic = cv2.resize(pic[Y0:Y0 + side, X0:X0 + side], (S, S), interpolation=cv2.INTER_AREA)
            text(pic, name, (12, 30), 0.8, (255, 255, 255), 2)
            if tid is not None:
                same = tid == before[name]
                text(pic, f"ID {tid % 1000}" + ("" if same or f < start else "   NEW ID: identity lost"), (12, 66), 0.9, (80, 255, 80) if same or f < start else (60, 60, 255), 2)
            else:
                text(pic, "out of sight" + (": the filter predicts the green ellipse" if ghost else ""), (12, 66), 0.7, (200, 200, 200), 2)
            panels.append(pic)
        bar = np.full((44, 2 * S + 8, 3), 12, np.uint8)
        text(bar, f"{seq}, person {person % 1000}: hidden for {(end - start + 1) * cam.dt:.1f} s.  Slowed 3x.  t = {(f - start) * cam.dt:+.1f} s", (12, 29), 0.65)
        writer.append_data(np.vstack([bar, np.hstack([panels[0], np.full((S, 8, 3), 12, np.uint8), panels[1]])])[:, :, ::-1])
    writer.close()
    print(out, "ID before:", before, "after:", {n: next((who[n][(person, f)] for f in range(end + 1, end + pad) if (person, f) in who[n]), None) for n in who})


if __name__ == "__main__":
    main(*sys.argv[1:8])
