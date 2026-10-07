"""Occlusion event viewer: one person, before -> hidden -> after, with MOT17 visibility and each tracker's identity.

Top: three panels (ground truth, SAM 3, SAM 3.1) cropped around the person. Ground truth shows the MOT17 box coloured
by visibility and the MOTS20 mask; tracker panels show the tracker mask matched to the person (mask IoU >= 0.5,
one-to-one per frame) with its ID. Bottom: visibility curve over time with a cursor, and one bar per tracker coloured
by the ID it gave the person in each frame (grey = not matched). An identity switch = the bar changes colour.

    python event_viewer.py --seq 02 --mot17-id 26 --frames 380 470 --out event_MOTS20-02_p26.mp4
"""
import argparse
import colorsys
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mu
from scipy.optimize import linear_sum_assignment

DATA = Path.home() / "Documents/Computervision_Research/Research_Data/MOTSChallenge/train"
TRK = Path(__file__).resolve().parents[1] / "pod_results/trackers/mots_challenge"
MOT17 = Path("/private/tmp/claude-501/-Users-sadeeqadeyemi-projects-AV-STACK/a2b9499e-95f2-414e-825d-513e4ddd3cc9/scratchpad/mot17/train")
FPS = {"02": 30, "05": 14, "09": 30, "11": 30}


def load_mots(path, keep_cls=2):
    out = defaultdict(dict)
    for line in open(path):
        f, tid, c, h, w, rle = line.split(" ", 5)
        if int(c) == keep_cls:
            out[int(f)][int(tid)] = {"size": [int(h), int(w)], "counts": rle.strip().encode()}
    return out


def id_color(tid):
    if tid is None:
        return (90, 90, 90)
    r, g, b = colorsys.hsv_to_rgb((tid * 0.618034) % 1, 0.8, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def vis_color(v):                                   # red (hidden) -> yellow -> green (visible), BGR
    v = float(np.clip(v, 0, 1))
    return (40, int(80 + 175 * min(1, 2 * v)), int(255 * min(1, 2 * (1 - v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="02")
    ap.add_argument("--mot17-id", type=int, default=26)
    ap.add_argument("--frames", type=int, nargs=2, default=[380, 470])
    ap.add_argument("--out", type=Path, default=Path("event_MOTS20-02_p26.mp4"))
    a = ap.parse_args()
    seq = f"MOTS20-{a.seq}"
    gt = load_mots(DATA / "instances_txt" / f"00{a.seq}.txt")
    trk = {name: load_mots(TRK / name / "data" / f"{seq}.txt") for name in ("SAM3", "SAM3.1")}
    mot = {}
    for line in open(MOT17 / f"MOT17-{a.seq}-FRCNN" / "gt" / "gt.txt"):
        r = line.strip().split(",")
        if int(r[1]) == a.mot17_id:
            mot[int(r[0])] = (float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[8]))
    gid = 2000 + a.mot17_id                          # MOTS20 person id (verified linked for this person)
    f0, f1 = a.frames
    frames = list(range(f0, f1 + 1))

    # per frame: tracker id matched to the person (Hungarian on mask IoU >= 0.5 over all people in the frame)
    matched = {name: {} for name in trk}
    for f in frames:
        g = gt.get(f, {})
        if gid not in g:
            continue
        gk = list(g)
        for name, t in trk.items():
            tf = t.get(f, {})
            if not tf:
                continue
            tk = list(tf)
            iou = np.asarray(mu.iou([tf[k] for k in tk], [g[k] for k in gk], [0] * len(gk))).reshape(len(tk), len(gk)).T
            for r, c in zip(*linear_sum_assignment(-iou)):
                if gk[r] == gid and iou[r, c] >= 0.5:
                    matched[name][f] = tk[c]

    # crop window around the person's MOT17 box (fixed for the clip)
    xs = [mot[f][0] for f in frames if f in mot] + [mot[f][0] + mot[f][2] for f in frames if f in mot]
    ys = [mot[f][1] for f in frames if f in mot] + [mot[f][1] + mot[f][3] for f in frames if f in mot]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w, half_h = 260, 200
    x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    PW, PH = 2 * half_w, 2 * half_h
    TH = 170
    W, H = 3 * PW, PH + 34 + TH
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", "10", "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(a.out)], stdin=subprocess.PIPE)
    fps = FPS[a.seq]
    for f in frames:
        img = cv2.imread(str(DATA / "images" / f"00{a.seq}" / f"{f:06d}.jpg"))
        panels = []
        for name in ("GT", "SAM3", "SAM3.1"):
            im = img.copy()
            if name == "GT":
                if f in gt and gid in gt[f]:
                    m = mu.decode(gt[f][gid]).astype(bool)
                    im[m] = (0.45 * im[m] + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
                if f in mot:
                    x, y, w, h, v = mot[f]
                    cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), vis_color(v), 3)
                title = f"Ground truth: MOT17 visibility {mot[f][4]:.2f}" if f in mot else "Ground truth"
                if f in mot and f not in gt.get(f, {}) and gid not in gt.get(f, {}):
                    title += "  (no MOTS20 mask)"
            else:
                tid = matched[name].get(f)
                if tid is not None:
                    m = mu.decode(trk[name][f][tid]).astype(bool)
                    im[m] = (0.45 * im[m] + 0.55 * np.array(id_color(tid))).astype(np.uint8)
                title = f"{'SAM 3' if name == 'SAM3' else 'SAM 3.1'}: " + (f"ID {tid % 1000}" if tid is not None else "no match")
            crop = im[y0:y0 + PH, x0:x0 + PW]
            crop = cv2.copyMakeBorder(crop, 0, PH - crop.shape[0], 0, PW - crop.shape[1], cv2.BORDER_CONSTANT)
            bar = np.full((34, PW, 3), 22, np.uint8)
            cv2.putText(bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            panels.append(np.vstack([bar, crop]))
        top = np.hstack(panels)
        # timeline: visibility curve + id bars
        tl = np.full((TH, W, 3), 28, np.uint8)
        L, R = 120, W - 30
        px = lambda fr: int(L + (fr - f0) / max(1, f1 - f0) * (R - L))
        vis_top, vis_bot = 12, 82
        cv2.putText(tl, "visibility", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(tl, (L, vis_bot - int(0.25 * (vis_bot - vis_top))), (R, vis_bot - int(0.25 * (vis_bot - vis_top))), (70, 70, 70), 1)
        pts = [(px(fr), int(vis_bot - mot[fr][4] * (vis_bot - vis_top))) for fr in frames if fr in mot]
        for p, q in zip(pts, pts[1:]):
            cv2.line(tl, p, q, (200, 200, 200), 2, cv2.LINE_AA)
        for k, name in enumerate(("SAM3", "SAM3.1")):
            yb = 96 + k * 34
            cv2.putText(tl, "SAM 3" if name == "SAM3" else "SAM 3.1", (10, yb + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            for fr in frames:
                cv2.rectangle(tl, (px(fr), yb), (px(fr + 1), yb + 24), id_color(matched[name].get(fr)), -1)
        cv2.line(tl, (px(f), 6), (px(f), TH - 6), (255, 255, 255), 2)
        cv2.putText(tl, f"frame {f}   t = {(f - f0) / fps:4.2f} s", (R - 230, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        enc.stdin.write(np.vstack([top, tl]).tobytes())
    enc.stdin.close(); enc.wait()
    print(a.out, {n: sorted(set(v.values())) for n, v in matched.items()})


if __name__ == "__main__":
    main()
