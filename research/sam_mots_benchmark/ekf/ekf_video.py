#!/usr/bin/env python3
"""HUD video of the EKF layer on a KITTI-MOTS sequence, 1920x1080.

    python ekf_video.py <tracker_in> <seq> <image_root> <out.mp4>

Camera (top): SAM's masks coloured by the ID the layer outputs, our lane, the car ahead in it, and for every object
a ring on the road where the filter expects it one second from now. A ring stays where it was drawn, so you watch
the object drive into it; when the second is up it turns green and the miss distance goes into that object's record.
Below: the road from above (with KITTI's own 3-D labels in grey, for checking), every filter's state, and the
identity timeline of the whole drive. Objects out of sight stay on screen in amber while the filter remembers them.

Everything is drawn relative to our own car, as the camera sees it. The layer is run first, frame by frame and on
past frames only; the video replays what it output. The 3-D labels are used to check the filter, never by it.
"""
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imageio
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw, ImageFont

import ekf_relink as E

W, H, CAM_H = 1920, 1080, 580     # picture; the camera view takes the top CAM_H rows
LANE = 1.75                       # m: half a lane; a car within this of our path is "in our lane"
ZMAX, BACK = 70.0, 5.0            # m: the view from above shows this far ahead and this far behind the camera
AHEAD, EVERY = 10, 5              # frames: each ring looks 1 s ahead; a new ring every 0.5 s
TRAIL = 30                        # frames of past positions drawn behind each object (3 s)
AA = cv2.LINE_AA
BG, PLATE, LINE, TEXT, DIM = (11, 13, 18), (21, 25, 33), (52, 60, 76), (236, 240, 245), (138, 148, 164)
AMBER, GREEN, WHITE, SKY, TRUE = (255, 184, 48), (72, 224, 128), (255, 255, 255), (64, 196, 255), (120, 128, 142)
PALETTE = [(64, 196, 255), (255, 112, 112), (186, 134, 255), (255, 138, 216), (64, 224, 208), (120, 160, 255),
           (236, 236, 150), (255, 150, 90)]


def font(size, weight="Regular", mono=False):
    try:
        f = ImageFont.truetype("/System/Library/Fonts/" + ("SFNSMono.ttf" if mono else "SFNS.ttf"), size)
        f.set_variation_by_name(weight)
        return f
    except OSError:                                   # not a Mac: whatever font Pillow ships
        return ImageFont.load_default(size)


CAP, SMALL, BODY, BOLD = font(12, "Semibold"), font(13), font(15, "Medium"), font(15, "Bold")
NUM, TINY, BIG, TITLE = font(14, "Medium", mono=True), font(11, "Medium"), font(34, "Bold"), font(20, "Bold")


class Text:
    """Text is queued and drawn last, in one pass, with a real font (OpenCV's own fonts are stick letters)."""

    def __init__(self):
        self.q = []

    def __call__(self, xy, s, f, col=TEXT, anchor="lm"):
        self.q.append((xy, s, f, col, anchor))

    def flush(self, img):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for xy, s, f, col, a in self.q:
            d.text(xy, s, font=f, fill=col, anchor=a)
        return np.array(im)


def ipt(p):
    return int(round(p[0])), int(round(p[1]))


def colour(i):
    return PALETTE[(i % 1000 + 3 * (i // 1000)) % len(PALETTE)]


def shade(img, x0, y0, x1, y1, col=BG, a=0.78):
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), img.shape[1]), min(int(y1), img.shape[0])
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - a) + np.array(col) * a).astype(np.uint8)


def dash(img, p, q, col, th=2, on=9, off=7):
    p, q = np.array(p, float), np.array(q, float)
    n = np.linalg.norm(q - p)
    for a in np.arange(0, n, on + off):
        cv2.line(img, ipt(p + (q - p) * a / n), ipt(p + (q - p) * min(a + on, n) / n), col, th, AA)


def corners(img, x, y, w, h, col, th=2, dashed=False):
    """A box drawn as four corner brackets (or dashed, for an object that is not seen right now)."""
    if dashed:
        for a, b in (((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)), ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))):
            dash(img, a, b, col, th, 7, 6)
        return
    n = max(6, min(w, h) / 4)
    for cx, sx in ((x, 1), (x + w, -1)):
        for cy, sy in ((y, 1), (y + h, -1)):
            cv2.line(img, ipt((cx, cy)), ipt((cx + sx * n, cy)), col, th, AA)
            cv2.line(img, ipt((cx, cy)), ipt((cx, cy + sy * n)), col, th, AA)


def chip(img, text, x, y, parts, f=BODY, taken=None):
    """A dark label with coloured parts [(text, colour)], its top-left near (x, y); moved up if it would cover another."""
    wd, ht = sum(f.getlength(s) for s, _ in parts) + 8 * (len(parts) - 1) + 14, f.size + 10
    x, y = min(max(x, 4), img.shape[1] - wd - 4), max(y, 4)
    for _ in range(4):
        if taken is None or not any(x < r[2] and r[0] < x + wd and y < r[3] and r[1] < y + ht for r in taken):
            break
        y -= ht + 2
    if taken is not None:
        taken.append((x, y, x + wd, y + ht))
    shade(img, x, y, x + wd, y + ht)
    cx = x + 7
    for s, col in parts:
        text((cx, y + ht / 2), s, f, col)
        cx += f.getlength(s) + 8


def panel(img, text, x0, y0, x1, y1, title, note=""):
    cv2.rectangle(img, (x0, y0), (x1, y1), PLATE, -1)
    text((x0 + 14, y0 + 17), title, CAP, DIM)
    text((x1 - 14, y0 + 17), note, SMALL, DIM, "rm")
    cv2.line(img, (x0, y0 + 32), (x1, y0 + 32), LINE, 1)


def labels(seq):
    """KITTI's own 3-D labels, made with the lidar: {frame: [(box x0 y0 x1 y1, X, Z, length, width, heading)]}."""
    out, p = defaultdict(list), E.DATA / f"x_data/data/gt/kitti/kitti_2d_box_train/label_02/{seq}.txt"
    for v in (line.split() for line in (p.read_text().splitlines() if p.exists() else [])):
        if v[2] != "DontCare":
            out[int(v[0])].append((np.array(v[6:10], float), float(v[13]), float(v[15]), float(v[12]), float(v[11]), float(v[16])))
    return out


def iou(b, g):                                          # b: x, y, width, height; g: x0, y0, x1, y1
    iw = min(b[0] + b[2], g[2]) - max(b[0], g[0])
    ih = min(b[1] + b[3], g[3]) - max(b[1], g[1])
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / (b[2] * b[3] + (g[2] - g[0]) * (g[3] - g[1]) - iw * ih)


def turn_rate(cam, k):
    """Our own turn rate now, rad/s, left positive (from the inertial unit's heading, one frame either side)."""
    a, b = max(k - 1, 0), min(k + 1, len(cam.yaw) - 1)
    return ((cam.yaw[b] - cam.yaw[a] + math.pi) % (2 * math.pi) - math.pi) / max((b - a) * cam.dt, cam.dt)


def ahead(cam, k, x, tau):
    """Where the filter expects an object relative to us in tau seconds: it keeps its velocity, we keep our speed
    and turn rate. (The same change of viewpoint as Ground.ego, over tau instead of one frame.)"""
    k = min(k, len(cam.vf) - 1)
    c, sn = math.cos(turn_rate(cam, k) * tau), math.sin(turn_rate(cam, k) * tau)
    return np.array([[c, sn], [-sn, c]]) @ (x[:2] + x[2:] * tau + np.array([cam.vl[k], -cam.vf[k]]) * tau)


def run(src, seq):
    """The layer on one sequence, frame by frame, past frames only. Returns what it output at every frame."""
    fr = E.read(Path(src) / "data" / f"{seq}.txt")
    h, w = next(iter(fr.values()))[0][2:4]
    cam, gt = E.Ground(seq, w, h), labels(seq)
    layer, frames, spent = E.Layer(cam), [], 0.0
    for k in range(min(fr), max(fr) + 1):
        rows = fr.get(k, [])
        t0 = time.perf_counter()
        dets = E.boxes(rows)
        ids = layer.step(k, dets)
        spent += time.perf_counter() - t0
        truth = {}                                      # the label under each mask, if one overlaps it well
        for tid, (_, _, b) in zip(ids, dets):
            g = max(gt[k], key=lambda g: iou(b, g[0]), default=None)
            if g is not None and iou(b, g[0]) > 0.5:
                truth[tid] = np.array(g[1:3])
        frames.append(dict(k=k, rows=rows, ids=ids, boxes=[d[2] for d in dets], sam={(d[1], d[0]) for d in dets}, truth=truth, gt=gt[k],
                           tracks={t["id"]: dict(cls=t["cls"], x=t["x"].copy(), P=t["P"][:2, :2].copy(), lost=t["lost"]) for t in layer.tracks},
                           relinks=[r for r in layer.relinks if r[0] == k]))
    return cam, frames, 1000 * spent / len(frames), w


def render(src, seq, image_root):
    cam, frames, ms, w = run(src, seq)
    s, n, dt = W / w, len(frames), cam.dt
    name = Path(src).name.replace("SAM", "SAM ")
    pix = lambda X, Z: cam.h(np.array([X, Z, 0, 0]))[:2] * s            # a point on the road -> the camera view

    state = {}                                                       # identity timeline: 1 seen, 2 out of sight
    for i, f in enumerate(frames):
        for tid, t in f["tracks"].items():
            state.setdefault(tid, np.zeros(n, np.uint8))[i] = 2 if t["lost"] else 1
    row, ends = {}, []
    for tid in sorted(state, key=lambda t: np.flatnonzero(state[t])[0]):
        a, b = np.flatnonzero(state[tid])[[0, -1]]
        r = next((r for r, e in enumerate(ends) if e + 15 < a), len(ends))
        ends[r:r + 1] = [b]
        row[tid] = r

    trail, foot, seen, size, rings, record = {}, {}, {}, {}, [], defaultdict(list)
    sam_ids, out_ids, restored, flash, pos_err, pred_err = set(), set(), [], {}, [], []
    for i, f in enumerate(frames):
        k, tr, text = f["k"], f["tracks"], Text()
        j = min(k, len(cam.vf) - 1)
        v_ego = cam.vf[j]
        rate = turn_rate(cam, j)
        path = lambda Z: -0.5 * rate / max(v_ego, 3.0) * Z * Z       # where our own car is heading (X at distance Z)
        ahead1 = lambda t: ahead(cam, k, t["x"], AHEAD * dt)

        sam_ids |= f["sam"]
        out_ids |= set(f["ids"])
        for r in f["relinks"]:
            restored.append((i, r[1], r[2]))
            flash[r[1]] = (i, r[2])
        trail = {tid: p for tid, p in trail.items() if tid in tr}
        foot = {tid: p for tid, p in foot.items() if tid in tr}
        seen = {tid: c for tid, c in seen.items() if tid in tr}
        for tid, t in tr.items():
            trail[tid] = (trail.get(tid, []) + [t["x"][:2]])[-TRAIL:]
            foot[tid] = (foot.get(tid, []) + [pix(*t["x"][:2])])[-TRAIL:] if t["x"][1] > 3 else []
            seen[tid] = seen.get(tid, 0) + (t["lost"] == 0)
        for tid, g in f["truth"].items():
            pos_err.append(np.linalg.norm(tr[tid]["x"][:2] - g))
        for r in rings:                                              # a ring's second is up: how far off was it?
            if r["at"] == i:
                if r["id"] in f["truth"]:
                    pred_err.append(np.linalg.norm(r["p"] - f["truth"][r["id"]]))
                if r["id"] in tr and not tr[r["id"]]["lost"]:
                    r["miss"] = np.linalg.norm(r["p"] - tr[r["id"]]["x"][:2])
                    record[r["id"]].append(r["miss"])
        rings = [r for r in rings if i - r["at"] < 4 and (i <= r["at"] or "miss" in r)]
        if i % EVERY == 0:
            rings += [dict(id=tid, at=i + AHEAD, p=ahead1(t)) for tid, t in tr.items()
                      if not t["lost"] and seen[tid] >= 10 and 3 < t["x"][1] < 60 and ahead1(t)[1] > 3]
        lead = min((tid for tid, t in tr.items() if t["cls"] == E.CAR and not t["lost"] and seen[tid] >= 10
                    and 2 < t["x"][1] < ZMAX and abs(t["x"][0] - path(t["x"][1])) < LANE), key=lambda tid: tr[tid]["x"][1], default=None)

        # ---- camera view
        bgr = cv2.imread(str(Path(image_root) / seq / f"{k:06d}.png"))
        view = (cv2.resize(bgr[:, :, ::-1], (W, CAM_H), interpolation=cv2.INTER_CUBIC) * 0.9).astype(np.uint8)
        zs = np.linspace(6, 60, 28)
        edge = lambda side: [pix(path(z) + side * LANE, z) for z in zs]
        ov = view.copy()
        cv2.fillPoly(ov, [np.int32(edge(-1) + edge(1)[::-1])], SKY)
        view = cv2.addWeighted(ov, 0.13, view, 0.87, 0)
        for side in (-1, 1):
            cv2.polylines(view, [np.int32(edge(side))], False, SKY, 1, AA)
        for tid, r in zip(f["ids"], f["rows"]):
            m = cv2.resize(mask_util.decode({"size": [r[2], r[3]], "counts": r[4].encode()}) * 255, (W, CAM_H), interpolation=cv2.INTER_LINEAR) > 127
            view[m] = (view[m] * 0.58 + np.array(colour(tid)) * 0.42).astype(np.uint8)
            cnt = [c for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c) > 40]
            cv2.drawContours(view, cnt, -1, colour(tid), 2, AA)
        for tid, t in tr.items():                                    # past positions, and the next second
            col = AMBER if t["lost"] else colour(tid)
            if len(foot[tid]) > 1:
                cv2.polylines(view, [np.int32(foot[tid])], False, col, 1, AA)
            p = ahead1(t)
            if seen[tid] >= 10 and t["x"][1] > 3 and p[1] > 3:
                dash(view, pix(*t["x"][:2]), pix(*p), col, 2)
        for r in rings:                                              # rings lie flat on the road, one car wide
            if r["p"][1] > 3:
                u, v = pix(*r["p"])
                rad = max(cam.fx * s * 0.9 / r["p"][1], 5)
                done = "miss" in r
                cv2.ellipse(view, ipt((u, v)), ipt((rad, max(rad * 0.3, 2))), 0, 0, 360, GREEN if done else colour(r["id"]), 3 if done else 2, AA)
        taken = [(0, 0, 470, 76), (W - 470, 0, W, 44)]
        for tid, box in sorted(zip(f["ids"], f["boxes"]), key=lambda p: -p[1][3]):
            t, (x0, y0, bw, bh) = tr[tid], box * s
            size[tid] = (bw, bh)
            new = tid in flash and i - flash[tid][0] < 20
            corners(view, x0, y0, bw, bh, GREEN if new else WHITE if tid == lead else colour(tid), 3 if new or tid == lead else 2)
            parts = [("LEAD", WHITE)] * (tid == lead) + [(f"{tid % 1000}", colour(tid)), (f"{t['x'][1]:.0f} m", TEXT)]
            if seen[tid] >= 10 and t["cls"] == E.CAR:
                parts.append((f"{3.6 * math.hypot(*t['x'][2:]):.0f} km/h", DIM))
            if new:
                parts.append((f"ID RESTORED after {flash[tid][1]:.1f} s", GREEN))
            chip(view, text, x0, y0 - 30, parts if bh > 34 or new or tid == lead else parts[:1], BOLD, taken)
        for tid, t in tr.items():                                    # out of sight: the filter keeps predicting
            u, v = pix(*t["x"][:2])
            if t["lost"] and t["x"][1] > 3 and tid in size and 0 <= u < W:
                bw, bh = size[tid]
                corners(view, u - bw / 2, v - bh, bw, bh, AMBER, 2, dashed=True)
                chip(view, text, u - bw / 2, v - bh - 30, [(f"{tid % 1000}", AMBER), (f"out of sight {t['lost'] * dt:.1f} s, predicted here", AMBER)], BOLD, taken)

        shade(view, 0, 0, 470, 76)
        text((18, 26), f"{name} + EKF identity layer", TITLE)
        text((18, 56), f"KITTI-MOTS {seq}   frame {i + 1} / {n}   {i * dt:5.1f} s", NUM, DIM)
        shade(view, W - 470, 0, W, 44)
        hidden = sum(1 for t in tr.values() if t["lost"])
        for x, label, val, col in ((W - 452, "TRACKED", len(tr) - hidden, TEXT), (W - 322, "OUT OF SIGHT", hidden, AMBER), (W - 162, "IDs RESTORED", len(restored), GREEN)):
            text((x, 22), label, CAP, DIM)
            text((x + CAP.getlength(label) + 8, 22), str(val), BOLD, col)
        shade(view, 16, CAM_H - 92, 190, CAM_H - 16)
        text((30, CAM_H - 76), "OUR SPEED", CAP, DIM)
        text((30, CAM_H - 44), f"{3.6 * v_ego:.0f}", BIG)
        text((36 + BIG.getlength(f"{3.6 * v_ego:.0f}"), CAM_H - 38), "km/h", BODY, DIM)
        shade(view, W / 2 - 330, CAM_H - 92, W / 2 + 330, CAM_H - 16)
        if lead is None:
            text((W / 2, CAM_H - 54), "No car ahead in our lane", BODY, DIM, "mm")
        else:
            x = tr[lead]["x"]
            text((W / 2 - 312, CAM_H - 76), f"CAR AHEAD IN OUR LANE   ID {lead % 1000}", CAP, DIM)
            cells = [("distance", f"{x[1]:.0f}", "m"), ("its speed", f"{3.6 * math.hypot(*x[2:]):.0f}", "km/h"),
                     ("time gap", f"{x[1] / v_ego:.1f}" if v_ego > 2 else "–", "s")]
            for c, (label, val, unit) in enumerate(cells):
                cx = W / 2 - 312 + 215 * c
                text((cx, CAM_H - 44), val, BIG)
                text((cx + BIG.getlength(val) + 6, CAM_H - 38), unit, BODY, DIM)
                text((cx + BIG.getlength(val) + 6, CAM_H - 56), label, TINY, DIM)

        img = np.full((H, W, 3), BG, np.uint8)
        img[:CAM_H] = view

        # ---- the road from above
        x0, y0, x1, y1 = 16, CAM_H + 14, 476, H - 38
        panel(img, text, x0, y0, x1, y1, "THE ROAD FROM ABOVE", "ring = expected in 1 s   grey = KITTI 3-D label")
        top, pw, ph = y0 + 33, x1 - x0, y1 - y0 - 33
        sc = ph / (ZMAX + BACK)
        bev = img[top:top + ph, x0:x1]
        m2p = lambda X, Z: (pw / 2 + X * sc, ph - (Z + BACK) * sc)
        cv2.fillPoly(bev, [np.int32([m2p(0, 0), m2p(-ZMAX * cam.cx / cam.fx, ZMAX), m2p(ZMAX * (w - cam.cx) / cam.fx, ZMAX)])], (27, 32, 42))
        zz = np.linspace(0, ZMAX, 36)
        cv2.fillPoly(bev, [np.int32([m2p(path(z) - LANE, z) for z in zz] + [m2p(path(z) + LANE, z) for z in zz[::-1]])], (30, 52, 70))
        for z in range(10, int(ZMAX) + 1, 10):
            cv2.line(bev, ipt(m2p(-99, z)), ipt(m2p(99, z)), LINE, 1)
            if z < ZMAX:
                text((x0 + 8, top + m2p(0, z)[1] - 9), f"{z} m", TINY, DIM)
        cv2.rectangle(bev, ipt(m2p(-0.9, 1.5)), ipt(m2p(0.9, -2.9)), WHITE, -1)
        text((x0 + m2p(0, 0)[0] + 12, top + m2p(0, -1.2)[1]), "us", TINY, TEXT)
        for box, gx, gz, gl, gw, ry in f["gt"]:                      # KITTI's labelled 3-D boxes, for checking by eye
            c, sn = math.cos(ry), math.sin(ry)
            quad = [m2p(gx + c * a + sn * b, gz - sn * a + c * b) for a, b in ((gl / 2, gw / 2), (gl / 2, -gw / 2), (-gl / 2, -gw / 2), (-gl / 2, gw / 2))]
            if gz < ZMAX + 5:
                cv2.polylines(bev, [np.int32(quad)], True, TRUE, 1, AA)
        for r in rings:
            done = "miss" in r
            cv2.circle(bev, ipt(m2p(*r["p"])), 7 if done else 6, GREEN if done else colour(r["id"]), 2, AA)
        for tid, t in sorted(tr.items(), key=lambda p: -p[1]["x"][1]):
            X, Z = t["x"][:2]
            if not -BACK < Z < ZMAX + 5:
                continue
            col, p = AMBER if t["lost"] else colour(tid), m2p(X, Z)
            if len(trail[tid]) > 1:
                cv2.polylines(bev, [np.int32([m2p(*q) for q in trail[tid]])], False, tuple(int(c * 0.6) for c in col), 2, AA)
            if seen[tid] >= 10:
                dash(bev, p, m2p(*ahead1(t)), col, 1, 5, 4)
            val, vec = np.linalg.eigh(t["P"] * sc * sc)              # uncertainty, 2 sigma (the picture's y axis points down)
            cv2.ellipse(bev, ipt(p), (int(min(2 * math.sqrt(max(val[1], 0)), 300)), int(min(2 * math.sqrt(max(val[0], 0)), 300))),
                        math.degrees(math.atan2(-vec[1, 1], vec[0, 1])), 0, 360, tuple(int(c * 0.7) for c in col), 1, AA)
            cv2.circle(bev, ipt(p), 5, col, 2 if t["lost"] else -1, AA)
            if tid == lead:
                cv2.circle(bev, ipt(p), 10, WHITE, 1, AA)
            if 0 < p[0] < pw - 30 and 12 < p[1] < ph:
                text((x0 + p[0] + 10, top + p[1] - 8), f"{tid % 1000}", TINY, col)

        # ---- every filter's state
        x0, x1 = 492, 1172
        panel(img, text, x0, y0, x1, y1, "WHAT EACH FILTER SAYS", "nearest first")
        cols = (34, 80, 146, 286, 366, 458, 574)
        for cx, label in zip(cols, ("ID", "TYPE", "DISTANCE", "SIDE", "SPEED", "1 s MISS", "STATE")):
            text((x0 + cx, y0 + 50), label, CAP, DIM)
        shown = sorted((tid for tid in tr if tr[tid]["x"][1] > 0), key=lambda tid: tr[tid]["x"][1])[:12]
        for r, tid in enumerate(shown):
            t, y = tr[tid], y0 + 78 + 30 * r
            if r % 2 == 0:
                cv2.rectangle(img, (x0 + 6, y - 14), (x1 - 6, y + 14), (26, 31, 40), -1)
            cv2.rectangle(img, (x0 + 14, y - 6), (x0 + 24, y + 6), AMBER if t["lost"] else colour(tid), -1)
            known = seen[tid] >= 10
            new = tid in flash and i - flash[tid][0] < 20
            miss = record.get(tid)
            cells = (f"{tid % 1000}", "car" if t["cls"] == E.CAR else "person", f"{t['x'][1]:5.1f} ± {math.sqrt(t['P'][1, 1]):4.1f} m",
                     f"{abs(t['x'][0]):4.1f} m {'R' if t['x'][0] > 0 else 'L'}", f"{3.6 * math.hypot(*t['x'][2:]):3.0f} km/h" if known else "  …",
                     f"{np.mean(miss):4.1f} m ({len(miss)})" if miss else "  –",
                     f"hidden {t['lost'] * dt:.1f} s" if t["lost"] else "ID RESTORED" if new else "LEAD" if tid == lead else "tracked")
            for cx, val in zip(cols, cells):
                state_col = AMBER if t["lost"] else GREEN if new else WHITE if tid == lead else DIM
                text((x0 + cx, y), val, NUM, state_col if cx == cols[-1] else TEXT)

        # ---- identity timeline
        x0, x1 = 1188, W - 16
        panel(img, text, x0, y0, x1, y1, "IDENTITY TIMELINE", f"{name} alone: {len(sam_ids)} IDs     with EKF: {len(out_ids)} IDs     {len(restored)} restored")
        left, right, top, bot = x0 + 16, x1 - 16, y0 + 48, y1 - 28
        rh = min(22.0, (bot - top) / max(len(ends), 1))
        tx = lambda q: left + (right - left) * q / max(n - 1, 1)
        for sec in range(0, int(n * dt) + 1, 5):
            cv2.line(img, ipt((tx(sec / dt), top - 6)), ipt((tx(sec / dt), bot)), (30, 35, 45), 1)
            text((tx(sec / dt), bot + 14), f"{sec} s", TINY, DIM, "mm")
        for tid, st in state.items():
            y, st = top + (row[tid] + 0.5) * rh, st[:i + 1].astype(int)
            cut = np.flatnonzero(np.diff(np.r_[0, st, 0]))             # frames where this identity's state changes
            for a, b in zip(cut[:-1], cut[1:]):
                if st[a]:
                    cv2.line(img, ipt((tx(a), y)), ipt((tx(b - 1) + 1, y)), colour(tid) if st[a] == 1 else AMBER, max(int(rh * 0.5), 3) if st[a] == 1 else 2)
            if st.any() and rh >= 14:
                text((tx(np.flatnonzero(st)[0]) - 3, y), f"{tid % 1000}", TINY, DIM, "rm")
        for ri, tid, _ in restored:
            cv2.drawMarker(img, ipt((tx(ri), top + (row[tid] + 0.5) * rh)), GREEN, cv2.MARKER_DIAMOND, 13, 2, AA)
        cv2.line(img, ipt((tx(i), top - 8)), ipt((tx(i), bot)), WHITE, 1)

        med = lambda v: f"{np.median(v):.1f} m" if v else "–"
        text((16, H - 19), f"Past frames only.   EKF layer: {ms:.1f} ms per frame on a laptop CPU.   Checked against KITTI's 3-D labels so far:  "
                           f"position off by {med(pos_err)} (median, {len(pos_err)} checks)   1 s prediction off by {med(pred_err)} ({len(pred_err)} checks)", SMALL, DIM)
        text((W - 16, H - 19), "amber = out of sight, still predicted     green diamond = ID given back", SMALL, DIM, "rm")
        yield text.flush(img)
    print(f"hand-backs {len(restored)}   position vs labels: median {med(pos_err)} (n={len(pos_err)})   1 s prediction: median {med(pred_err)} (n={len(pred_err)})")


def main(src, seq, image_root, out):
    wr = imageio.get_writer(out, fps=10, codec="libx264", quality=None, macro_block_size=1,
                            output_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"])
    for img in render(src, seq, image_root):
        wr.append_data(img)
    wr.close()
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
