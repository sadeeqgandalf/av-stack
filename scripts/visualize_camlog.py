"""Visualise one camera-log frame: what ground truth the simulator provides, and how a
predictor (the stack's tracks, SAM 3 / SAM 3.1) lines up with it.

    PYTHONPATH=. .venv/bin/python scripts/visualize_camlog.py logs/crosswalk_multiple_threat_seed0 \\
        --camera front --frame auto --out results/figures/gt_frame.png

Panels: proxy RGB | semantic (CARLA colours) | depth | instance ids with boxes.
Boxes: green = ground truth (amodal 3D box projected; dashed when "don't care"),
orange = stack track matched (IoU >= 0.5), red = stack track unmatched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_perception as sp  # noqa: E402

from avstack.sim.render import colorize_semantic  # noqa: E402


def font(size):
    for f in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dashed_rect(d: ImageDraw.ImageDraw, box, color, width=2, dash=6):
    x0, y0, x1, y1 = box
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash), 1)
        for k in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            q = (a[0] + (b[0] - a[0]) * min(k + 1, n) / n, a[1] + (b[1] - a[1]) * min(k + 1, n) / n)
            d.line([p, q], fill=color, width=width)


def pick_frame(gt: dict) -> int:
    """The frame with the most scored ground-truth objects (ties: pedestrians first)."""
    def key(f):
        items = [x for x in gt[f] if not x["ignore"]]
        return (len(items), sum(x["cls"] == "pedestrian" for x in items))
    return max(gt, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--frame", default="auto")
    ap.add_argument("--out", type=Path, default=Path("results/figures/gt_frame.png"))
    a = ap.parse_args()

    gt_all = sp.load_gt(a.log, a.camera, 150, amodal=True)
    f = pick_frame(gt_all) if a.frame == "auto" else int(a.frame)
    frames = {json.loads(line)["i"]: json.loads(line) for line in (a.log / "frames.jsonl").read_text().splitlines()}
    fr = frames[f]
    base = a.log / a.camera
    name = f"{f:05d}"
    rgb = np.array(Image.open(base / "rgb" / f"{name}.jpg"))
    sem = colorize_semantic(np.array(Image.open(base / "semantic" / f"{name}.png")))
    depth_cm = np.array(Image.open(base / "depth" / f"{name}.png")).astype(np.float32)
    inst = np.array(Image.open(base / "instance" / f"{name}.png"))
    d = np.where(depth_cm > 0, depth_cm / 100.0, 200.0)
    dv = (255 * (1 - np.clip(np.log1p(d) / np.log1p(200.0), 0, 1))).astype(np.uint8)
    rng = np.random.default_rng(3)
    lut = rng.integers(60, 255, (65536, 3), dtype=np.uint8)
    lut[0] = 0
    inst_rgb = lut[inst]

    H, W = rgb.shape[:2]
    pad, head = 12, 34
    canvas = Image.new("RGB", (2 * W + 3 * pad, 2 * (H + head) + 3 * pad + 40), (18, 18, 20))
    panels = [(rgb, "1  Proxy RGB (what a camera would see; boxes on a plane)"),
              (sem, "2  Semantic ground truth (CARLA tags: road, sidewalk, car, pedestrian …)"),
              (np.stack([dv] * 3, -1), "3  Depth ground truth (planar, cm; near = bright)"),
              (inst_rgb, "4  Instance ids + boxes (green GT, orange matched track, red unmatched)")]
    fnt, small = font(15), font(12)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), f"{a.log.name} · camera {a.camera} · frame {f} (t = {fr['t']:.1f} s)", fill=(255, 255, 255),
              font=font(17))
    for k, (img, title) in enumerate(panels):
        x = pad + (k % 2) * (W + pad)
        y = 40 + pad + (k // 2) * (H + head + pad)
        draw.text((x, y), title, fill=(195, 194, 183), font=fnt)
        canvas.paste(Image.fromarray(img), (x, y + head))
    # panel 4 is a zoomed crop around the objects (at 30 m+ they are only a few dozen px wide)
    gt = gt_all[f]
    tracks = sp.project_tracks(a.log, a.camera).get(f, [])
    boxes = [g["box"] for g in gt] + [t["box"] for t in tracks]
    if boxes:
        B = np.array(boxes, float)
        bx0, by0, bx1, by1 = B[:, 0].min(), B[:, 1].min(), B[:, 2].max(), B[:, 3].max()
    else:
        bx0, by0, bx1, by1 = 0, 0, W, H
    cw = max(bx1 - bx0, (by1 - by0) * W / H) * 1.6 + 20          # keep the panel's aspect ratio
    ch = cw * H / W
    cx0 = float(np.clip((bx0 + bx1) / 2 - cw / 2, 0, max(W - cw, 0)))
    cy0 = float(np.clip((by0 + by1) / 2 - ch / 2, 0, max(H - ch, 0)))
    zoom = W / cw
    crop = Image.fromarray(inst_rgb).crop((int(cx0), int(cy0), int(cx0 + cw), int(cy0 + ch))).resize(
        (W, H), Image.NEAREST)
    x0, y0 = pad + W + pad, 40 + pad + (H + head + pad) + head
    canvas.paste(crop, (x0, y0))
    draw.rectangle([x0, y0 - head, x0 + W, y0 - 4], fill=(18, 18, 20))
    draw.text((x0, y0 - head), f"4  Instance ids, zoomed {zoom:.1f}x (green GT · orange matched track · red unmatched)",
              fill=(195, 194, 183), font=fnt)
    # the zoom region, outlined on the RGB panel
    rx, ry = pad, 40 + pad + head
    draw.rectangle([rx + cx0, ry + cy0, rx + cx0 + cw, ry + cy0 + ch], outline=(255, 255, 255), width=1)
    iou = sp.iou_matrix(np.array([g["box"] for g in gt], float).reshape(-1, 4),
                        np.array([t["box"] for t in tracks], float).reshape(-1, 4))
    matched_tracks = set()
    if iou.size:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-iou)
        matched_tracks = {j for i, j in zip(r, c) if iou[i, j] >= 0.5}
    shift = lambda b: [x0 + (b[0] - cx0) * zoom, y0 + (b[1] - cy0) * zoom,          # noqa: E731
                       x0 + (b[2] - cx0) * zoom, y0 + (b[3] - cy0) * zoom]
    desc = {o["id"]: o for o in fr["objects"]}
    label_font = font(14)
    for k, g in enumerate(gt):
        b = shift(g["box"])
        if g["ignore"]:
            dashed_rect(draw, b, (12, 163, 12), width=3)
        else:
            draw.rectangle(b, outline=(12, 163, 12), width=3)
        vis = int((inst == g["id"] + 1).sum())
        draw.text((b[0] + 3, b[3] + 4 + 17 * (k % 2)),
                  f"GT #{g['id']} {g['cls']} · {vis} px{' · don’t care' if g['ignore'] else ''}",
                  fill=(120, 220, 120), font=label_font)
    for j, t in enumerate(tracks):
        col = (236, 131, 90) if j in matched_tracks else (208, 59, 59)
        b = shift(t["box"])
        draw.rectangle(b, outline=col, width=2)
        draw.text((b[0] + 3, max(b[1] - 18 - 17 * (j % 2), y0)), f"track {t['id']}", fill=col, font=label_font)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    print(f"frame {f}: {len(gt)} GT objects ({sum(not g['ignore'] for g in gt)} scored), "
          f"{len(tracks)} stack tracks in view, {len(matched_tracks)} matched -> {a.out}")
    for g in gt:
        o = desc[g["id"]]
        print(f"  GT #{g['id']:>2} {g['cls']:10s} '{o['description']}'  box={[round(v) for v in g['box']]}"
              f"{'  (dont care)' if g['ignore'] else ''}")


if __name__ == "__main__":
    main()
