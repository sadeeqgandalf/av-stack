"""Side-by-side review: ground truth (top) vs a tracker's masks and IDs (bottom) on the CARLA front camera.
A tracker ID keeps its colour, so an identity switch shows as a colour change on the same object.

    python sam_video.py runs/ped_behind_van runs/ped_behind_van/sam/SAM3/data.txt "SAM 3"
"""
import colorsys
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util


def load(path):
    out = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        f, tid, cls, h, w, rle = line.split(" ", 5)
        out[int(f)][int(tid)] = mask_util.decode({"size": [int(h), int(w)], "counts": rle.strip().encode()}).astype(bool)
    return out


def color(tid):
    """Distinct hue per ID; pedestrians start half-way round the colour wheel from cars."""
    r, g, b = colorsys.hsv_to_rgb(((tid % 1000) * 0.618034 + (0.5 if tid >= 2000 else 0.0)) % 1, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(img, masks, title, name):
    img = img.copy()
    for tid, m in masks.items():
        c = color(tid)
        img[m] = (0.45 * img[m] + 0.55 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, c, 1, cv2.LINE_AA)
        ys, xs = np.nonzero(m)
        lbl = f"{name} {'ped' if tid >= 2000 else 'car'} {tid % 1000}"
        cv2.putText(img, lbl, (int(xs.min()), max(40, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (int(xs.min()), max(40, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return img


seq, trk_path, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
gt, trk = load(seq / "label_02.txt"), load(trk_path)
frames = sorted((seq / "full" / "rgb").glob("*.png"))
h, w = cv2.imread(str(frames[0])).shape[:2]
out = trk_path.parent / f"{label.replace(' ', '')}_vs_groundtruth.mp4"
enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{2 * h}", "-r", "30",
                        "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for i, fp in enumerate(frames):
    img = cv2.imread(str(fp))
    top = draw(img, gt.get(i, {}), f"Ground truth (CARLA)   frame {i}", "GT")
    bot = draw(img, trk.get(i, {}), f"{label}: masks and IDs (same colour = same identity)", "ID")
    enc.stdin.write(np.vstack([top, bot]).tobytes())
enc.stdin.close(); enc.wait()
print(out)
