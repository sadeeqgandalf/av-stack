#!/usr/bin/env python3
"""Render a tracker's MOTS output (the exact file TrackEval scored) as an H.264 video.

Masks are coloured by track identity: an object keeps its colour and number as long as
the tracker keeps its identity, so identity switches are visible as colour changes.

    python render_video.py --tracker-file trackers/kitti_mots/SAM3.1/data/0016.txt \
        --frames .../image_02/0016 --frame-offset 0 --fps 10 --scale 1.5 \
        --title "SAM 3.1 · KITTI-MOTS val 0016" --out videos/SAM3.1_KITTI-MOTS_0016.mp4
"""
from __future__ import annotations

import argparse
import colorsys
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

CLASS_NAME = {1: "car", 2: "ped"}


def ascii_text(s: str) -> str:
    """Hershey fonts are ASCII-only."""
    return s.replace("\u00b7", "|").replace("\u2013", "-").replace("\u2014", "-").encode("ascii", "replace").decode()


def id_color(tid: int) -> tuple[int, int, int]:
    """Distinct, stable colour per identity (golden-ratio hue walk), BGR."""
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def load_tracks(path: Path, offset: int) -> dict[int, list]:
    per_frame = defaultdict(list)
    for line in path.read_text().splitlines():
        f, tid, cls, h, w, rle = line.split(" ", 5)
        per_frame[int(f) - offset].append((int(tid), int(cls), {"size": [int(h), int(w)], "counts": rle.encode()}))
    return per_frame


def draw_label(img, text, x, y, color, scale):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, int(round(1.2 * scale)))
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    x = int(np.clip(x, 0, img.shape[1] - tw - 8))
    y = int(np.clip(y, tht + 8, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - tht - 7), (x + tw + 8, y + 2), (20, 20, 22), -1)
    cv2.rectangle(img, (x, y - tht - 7), (x + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 6, y - 3), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker-file", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--frame-offset", type=int, default=0, help="first frame number in the MOTS file")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.title, a.subtitle = ascii_text(a.title), ascii_text(a.subtitle)

    frames = sorted(p for p in a.frames.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    tracks = load_tracks(a.tracker_file, a.frame_offset)
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    W, H = int(round(w0 * a.scale / 2) * 2), int(round(h0 * a.scale / 2) * 2)
    header = int(round(46 * max(a.scale, 1.0) * (1 if w0 < 1500 else 1.25) / 2) * 2)   # H.264 needs even dims
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + header}",
         "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
    seen: set[int] = set()
    ts = w0 / 1242.0 * a.scale if w0 < 1500 else 1.9          # text scale
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        overlay = img.copy()
        items = tracks.get(i, [])
        labels = []
        for tid, cls, rle in items:
            m = mask_util.decode(rle).astype(bool)
            if not m.any():
                continue
            col = id_color(tid)
            overlay[m] = (0.45 * np.array(col) + 0.55 * overlay[m]).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, col, max(1, int(round(1.5 * a.scale))), cv2.LINE_AA)
            ys, xs = np.nonzero(m)
            labels.append((f"{CLASS_NAME.get(cls, cls)} #{tid % 1000}", xs.min(), ys.min(), col))
            seen.add(tid)
        frame = cv2.resize(overlay, (W, H), interpolation=cv2.INTER_AREA if a.scale < 1 else cv2.INTER_CUBIC)
        for text, x, y, col in labels:
            draw_label(frame, text, x * W / w0, y * H / h0 - 2, col, ts)
        bar = np.full((header, W, 3), (24, 24, 26), np.uint8)
        f1 = 0.62 * ts * 1.15
        cv2.putText(bar, a.title, (14, int(header * 0.44)), cv2.FONT_HERSHEY_SIMPLEX, f1, (255, 255, 255),
                    max(1, int(1.6 * ts)), cv2.LINE_AA)
        info = f"{a.subtitle}   frame {i + 1}/{len(frames)}   tracks in frame: {len(labels)}   identities so far: {len(seen)}"
        cv2.putText(bar, info, (14, int(header * 0.84)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ts * 1.15, (195, 194, 183),
                    max(1, int(1.1 * ts)), cv2.LINE_AA)
        enc.stdin.write(np.vstack([bar, frame]).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"{a.out}: {len(frames)} frames @ {a.fps} fps, {W}x{H + header}, {len(seen)} identities")


if __name__ == "__main__":
    main()
