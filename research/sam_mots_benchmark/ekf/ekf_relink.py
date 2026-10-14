#!/usr/bin/env python3
"""An EKF identity layer on top of SAM's tracking output.

Every object SAM reports gets a small filter that predicts where it will be in the next frame. While SAM keeps an
object's ID, the filter just follows it. When an object disappears, its filter keeps predicting. When SAM then
reports a NEW ID, we ask one question before accepting it: is this someone we lost, showing up where their filter
said they would be? If yes, the old ID is given back. Only past frames are used, so it can run live.

    python ekf_relink.py kitti <tracker_in> <tracker_out> [--pixel] [--scratch]
    python ekf_relink.py mots  <tracker_in> <tracker_out> [--scratch]

<tracker_in>/data/<seq>.txt is the MOTS text format (frame id class h w rle). --scratch erases SAM's IDs first, so the
filter alone decides every identity. --pixel runs KITTI with the image-plane model used for MOTS20 (no calibration).
Method and every value below: README.md in this folder.
"""
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import mahalanobis
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

DATA = Path(__file__).parent / "_data"
CAR, PED = 1, 2

# ---- every number the filter uses. Noise values come from physics; MEMORY, CUT and SHRUNK were chosen on the
# 12 KITTI-MOTS training sequences outside the val split (dev_fragments.py), which are not scored.
SIG_ACC = {CAR: 3.0, PED: 1.0}    # m/s^2: how hard an object may speed up, brake or turn between frames (process noise)
SIG_V0 = {CAR: 10.0, PED: 1.5}    # m/s: how unsure we are of a new object's speed (it starts at 0 +/- this)
HEIGHT = {CAR: 1.5, PED: 1.7}     # m: typical height. Turns how tall a mask looks into how far away it is
PIX0, PIXREL = 2.0, 0.10          # measurement noise: 2 px + 10 % of the mask's box side (masks jitter more when big)
SIZE0, SIZEREL = 1.0, 0.08        # noise of the mask's height as a distance cue (KITTI): 1 px + 8 %, because real cars
                                  # are not all HEIGHT tall (labelled cars on the dev sequences: 1.39 to 1.66 m)
GATE = {2: 11.83, 3: 14.16}       # chi-square for 2 or 3 numbers measured: 99.73 % of true matches fall inside it
MEMORY = 1.0                      # s: how long a lost object is remembered. After about a second the search area is
                                  # so large that newcomers fit it as often as the real owner (dev set: 1 s beat 2 and 3 s)
CUT, SHRUNK = 5.0, 0.8            # a box cut by the image edge, or under 80 % of its remembered height (partly hidden),
                                  # no longer shows the true centre or road contact: 5x the noise in that direction
CAM_HEIGHT = 1.65                 # m: KITTI camera above the road (from the KITTI sensor setup)


class Pixel:
    """No calibration (MOTS20): the state is in pixels, [u, v, du/dt, dv/dt]. The camera measures (u, v) directly, so
    the measurement function is the identity and the EKF is an ordinary Kalman filter here."""

    size = False                                    # the mask's height is not used as a measurement here

    def __init__(self, fps, width, height):
        self.dt, self.W, self.H = 1.0 / fps, width, height

    def z(self, box):                               # what we measure from a mask: the centre of its box
        return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])

    def scale(self, cls, hpx):                      # pixels per metre at this object
        return hpx / HEIGHT[cls]

    def h(self, x, cls=CAR):
        return x[:2].copy()

    def jac(self, x, cls=CAR):
        return np.eye(2, 4)

    def birth(self, z, R, cls=CAR):
        return np.array([z[0], z[1], 0, 0.0]), R

    def ego_terms(self, k):                         # no calibration, no own-motion data: nothing to remove
        return np.eye(4), np.zeros(4)

    def ego(self, k, x, P):
        return x, P


class Ground:
    """KITTI: the state is in metres on the road, seen from above: [X right, Z ahead, vX, vZ], velocity over the ground.
    The camera measures three numbers per mask: its centre column, the row where it meets the road, and how tall it
    looks. All three depend on 1/Z, so the measurement is nonlinear: an EKF. Two of them say how far the object is.
    Where it meets the road is exact on a flat road but off by 25 % beyond 45 m on real ones (dev sequences against
    KITTI's 3-D labels); how tall it looks is off by 7 % there, because cars differ less than roads do."""
    size = True

    def __init__(self, seq, width, height):
        self.dt, self.W, self.H = 0.1, width, height
        P2 = next(l for l in (DATA / f"x_data_tracking_calib/training/calib/{seq}.txt").read_text().splitlines() if l.startswith("P2"))
        p = [float(v) for v in P2.split()[1:]]
        self.fx, self.cx, self.fy, self.cy = p[0], p[2], p[5], p[6]
        ox = np.loadtxt(DATA / f"x_data_tracking_oxts/training/oxts/{seq}.txt")
        self.yaw, self.vf, self.vl = ox[:, 5], ox[:, 8], ox[:, 9]   # heading (rad, left turn positive), speed ahead, speed left

    def z(self, box):                               # centre column, row where it meets the road, height in pixels
        return np.array([box[0] + box[2] / 2, box[1] + box[3], box[3]])

    def scale(self, cls, hpx):
        return 1.0

    def h(self, x, cls=CAR):
        Z = max(x[1], 0.5)
        return np.array([self.fx * x[0] / Z + self.cx, self.fy * CAM_HEIGHT / Z + self.cy, self.fy * HEIGHT[cls] / Z])

    def jac(self, x, cls=CAR):
        X, Z = x[0], max(x[1], 0.5)
        return np.array([[self.fx / Z, -self.fx * X / Z ** 2, 0, 0], [0, -self.fy * CAM_HEIGHT / Z ** 2, 0, 0],
                         [0, -self.fy * HEIGHT[cls] / Z ** 2, 0, 0]])

    def birth(self, z, R, cls=CAR):
        """First position from one mask: the two distance cues, each weighted by how sure it is."""
        dv = max(z[1] - self.cy, 1.0)               # ponytail: at or above the horizon the road cue says nothing
        Zr, Zs = self.fy * CAM_HEIGHT / dv, self.fy * HEIGHT[cls] / max(z[2], 1.0)
        wr, ws = 1 / (Zr * math.sqrt(R[1, 1]) / dv) ** 2, 1 / (Zs * math.sqrt(R[2, 2]) / max(z[2], 1.0)) ** 2
        Z, var = (wr * Zr + ws * Zs) / (wr + ws), 1 / (wr + ws)
        J = np.array([[Z / self.fx, (z[0] - self.cx) / self.fx], [0, 1.0]])       # d(X, Z) / d(u, Z)
        return np.array([(z[0] - self.cx) * Z / self.fx, Z, 0, 0.0]), J @ np.diag([R[0, 0], var]) @ J.T

    def ego_terms(self, k):
        """Our own car moved between frame k-1 and k. Returns G (its turn) and d (its shift) such that a point's state in
        the car's new position is G (x + d). ponytail: speed is read at the inertial unit, not the camera (1 m apart);
        matters only in sharp turns."""
        if k <= 0 or k >= len(self.yaw):
            return np.eye(4), np.zeros(4)
        a = (self.yaw[k] - self.yaw[k - 1] + math.pi) % (2 * math.pi) - math.pi
        c, s = math.cos(a), math.sin(a)
        A = np.array([[c, s], [-s, c]])
        G = np.zeros((4, 4)); G[:2, :2] = A; G[2:, 2:] = A
        return G, np.array([self.vl[k - 1] * self.dt, -self.vf[k - 1] * self.dt, 0, 0])

    def ego(self, k, x, P):
        G, d = self.ego_terms(k)
        return G @ (x + d), G @ P @ G.T


def predict(cam, k, t):
    """Constant velocity, then our own car's motion. The model is defined here; filterpy does the maths:
    x = F x + B u, P = F P F' + Q, with F = G Fcv, Q = G Qcv G', B = G and u = d (see ego_terms)."""
    dt = cam.dt
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
    Q = np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]]) * (SIG_ACC[t["cls"]] * cam.scale(t["cls"], t["hpx"])) ** 2
    G, d = cam.ego_terms(k)
    kf = t["kf"]
    kf.F, kf.Q, kf.B = G @ F, G @ Q @ G.T, G
    kf.predict(u=d)
    t["x"], t["P"] = kf.x, kf.P
    t["zh"], t["H"] = cam.h(t["x"], t["cls"]), cam.jac(t["x"], t["cls"])


def noise(cam, box, hpx):
    su, sv, sh = PIX0 + PIXREL * box[2], PIX0 + PIXREL * box[3], SIZE0 + SIZEREL * box[3]
    if box[0] <= 1 or box[0] + box[2] >= cam.W - 1:
        su *= CUT
    if box[1] <= 1 or box[1] + box[3] >= cam.H - 1 or box[3] < SHRUNK * hpx:
        sv, sh = sv * CUT, sh * CUT                    # cut or partly hidden: neither the road contact nor the height is whole
    return np.diag([su ** 2, sv ** 2, sh ** 2] if cam.size else [su ** 2, sv ** 2])


def innovation(cam, t, box):
    """How far a detection is from where the filter expected it, in units of the filter's own uncertainty."""
    y = cam.z(box) - t["zh"]
    S = t["H"] @ t["P"] @ t["H"].T + noise(cam, box, t["hpx"])   # the S filterpy would form in update(); needed before it
    return y, S, float(mahalanobis(y, np.zeros_like(y), S)) ** 2


def update(cam, t, box):
    """filterpy's EKF update (Joseph form): K = P H' S^-1, x += K (z - h(x)), P = (I-KH) P (I-KH)' + K R K'."""
    kf = t["kf"]
    kf.update(cam.z(box), cam.jac, cam.h, R=noise(cam, box, t["hpx"]), args=(t["cls"],), hx_args=(t["cls"],))
    t["x"], t["P"] = kf.x, kf.P
    t["hpx"] = max(box[3], t["hpx"] * 0.8 ** cam.dt)   # remembered height: follows growth at once, forgets 20 % per second
    t["lost"] = 0


class Layer:
    def __init__(self, cam, relink=True):
        self.cam, self.relink = cam, relink
        self.tracks, self.of_sam, self.next_id, self.relinks = [], {}, defaultdict(int), []

    def step(self, k, dets, keep_ids=True):
        """dets: [(sam_id, cls, box xywh)]. Returns the output ID for each detection."""
        cam, out = self.cam, [None] * len(dets)
        for t in self.tracks:
            predict(cam, k, t); t["lost"] += 1
        todo = []
        for j, (sid, cls, box) in enumerate(dets):                 # 1. SAM still holds this ID: follow it
            t = self.of_sam.get((cls, sid)) if keep_ids else None
            if t is None:
                todo.append(j)
            elif t["alive"]:
                update(cam, t, box); out[j] = t["id"]
            else:                                                  # SAM kept the ID longer than our memory: honour it
                self._start(t, box); out[j] = t["id"]
        free = [t for t in self.tracks if t["lost"] > 0 and self.relink
                and 0 <= t["zh"][0] < cam.W and 0 <= t["zh"][1] < cam.H]   # lost, and expected inside the picture
        if todo and free:                                          # 2. a new ID: is it someone we lost?
            cost = np.full((len(todo), len(free)), 1e6)
            for a, j in enumerate(todo):
                for b, t in enumerate(free):
                    if t["cls"] == dets[j][1]:
                        _, S, d2 = innovation(cam, t, dets[j][2])
                        if d2 <= GATE[len(t["zh"])]:
                            cost[a, b] = d2 + math.log(np.linalg.det(S))   # closest, with ties going to the surer track
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    j, t = todo[a], free[b]
                    if t["lost"] > 1:
                        self.relinks.append((k, t["id"], (t["lost"] - 1) * cam.dt))
                    self.of_sam = {s: u for s, u in self.of_sam.items() if u is not t}
                    self.of_sam[(dets[j][1], dets[j][0])] = t
                    update(cam, t, dets[j][2]); out[j] = t["id"]
        for j in todo:                                             # 3. nobody we know: a new object
            if out[j] is None:
                sid, cls, box = dets[j]
                self.next_id[cls] += 1
                t = dict(id=cls * 1000 + self.next_id[cls], cls=cls)
                self._start(t, box); self.of_sam[(cls, sid)] = t; out[j] = t["id"]
        for t in self.tracks:
            t["alive"] = t["lost"] * cam.dt <= MEMORY
        self.tracks = [t for t in self.tracks if t["alive"]]
        if not keep_ids:
            self.of_sam = {}
        return out

    def _start(self, t, box):
        """A new filter from one detection: position from the camera, speed unknown."""
        cam, cls = self.cam, t["cls"]
        x, Ppos = cam.birth(cam.z(box), noise(cam, box, box[3]), cls)
        P = np.diag([0, 0, 1, 1.0]) * (SIG_V0[cls] * cam.scale(cls, box[3])) ** 2
        P[:2, :2] = Ppos
        kf = ExtendedKalmanFilter(dim_x=4, dim_z=len(cam.z(box)))
        kf.x, kf.P = x.astype(float), P
        t.update(kf=kf, x=kf.x, P=kf.P, hpx=box[3], lost=0, alive=True, zh=cam.h(x, cls), H=cam.jac(x, cls))
        self.tracks.append(t)


def read(path):
    """{frame: [(id, class, h, w, rle)]} for cars and pedestrians."""
    fr = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        f, tid, cls, h, w, rle = line.split(" ", 5)
        if int(cls) in (CAR, PED):
            fr[int(f)].append((int(tid), int(cls), int(h), int(w), rle.strip()))
    return fr


def main_box(h, w, rle):
    """Box (x, y, width, height) of the mask's largest connected part. SAM masks often carry stray specks far from
    the object (on MOTS20: 12 % of SAM 3 masks, 38 % of SAM 3.1 masks), which drag a plain bounding box off it."""
    obj = {"size": [h, w], "counts": rle.encode()}
    x, y, bw, bh = (int(v) for v in mask_util.toBbox(obj))
    lab, n = ndimage.label(mask_util.decode(obj)[y:y + bh, x:x + bw], structure=np.ones((3, 3)))
    if n <= 1:
        return np.array([x, y, bw, bh], float)
    rows, cols = ndimage.find_objects(lab)[np.bincount(lab.ravel())[1:].argmax()]
    return np.array([x + cols.start, y + rows.start, cols.stop - cols.start, rows.stop - rows.start], float)


def boxes(rows):
    return [(tid, cls, main_box(h, w, rle)) for tid, cls, h, w, rle in rows]


def camera(bench, seq, h, w, pixel=False):
    if bench == "kitti":
        return Pixel(10, w, h) if pixel else Ground(seq, w, h)
    ini = (DATA / f"x_data/data/gt/mot_challenge/MOTS-train/{seq}/seqinfo.ini").read_text()
    return Pixel(float(ini.split("frameRate=")[1].split()[0]), w, h)


def run_file(bench, src, dst, pixel=False, scratch=False, relink=True, on_frame=None):
    fr = read(src)
    if not fr:
        Path(dst).write_text(""); return 0, 0, 0.0
    h, w = next(iter(fr.values()))[0][2:4]
    layer = Layer(camera(bench, Path(src).stem, h, w, pixel), relink)
    lines, t0, first = [], time.perf_counter(), min(fr)
    for k in range(first, max(fr) + 1):
        rows = fr.get(k, [])
        ids = layer.step(k if bench == "kitti" else k - first, boxes(rows), keep_ids=not scratch)
        lines += [f"{k} {i} {r[1]} {r[2]} {r[3]} {r[4]}" for i, r in zip(ids, rows)]
        if on_frame:
            on_frame(k, layer, rows, ids)
    dt = time.perf_counter() - t0
    Path(dst).write_text("\n".join(lines) + "\n")
    return len(layer.relinks), max(fr) + 1 - first, dt


def self_check():
    cam = Ground("0008", 1242, 375)
    x = np.array([2.0, 20.0, 0, 0])
    b, _ = cam.birth(cam.h(x), np.eye(3))
    assert np.allclose(b, x, atol=1e-6), "pixel -> road -> pixel must return the same point"
    tall = cam.h(x) + np.array([0, 0, 25.0])                                        # the mask looks taller than the road contact implies
    assert cam.birth(tall, np.eye(3))[0][1] < 19 < cam.birth(tall, np.diag([1, 1e-4, 1.0]))[0][1] <= 20, \
        "a taller mask means a nearer object, unless the road contact is trusted far more"
    m = np.zeros((50, 80), np.uint8); m[10:30, 20:40] = 1; m[45, 70] = 1            # an object and one stray pixel
    rle = mask_util.encode(np.asfortranarray(m))["counts"].decode()
    assert main_box(50, 80, rle).tolist() == [20, 10, 20, 20], "a stray speck must not move the measured box"
    num = np.array([(cam.h(x + e) - cam.h(x - e)) / 2e-5 for e in np.eye(4) * 1e-5]).T
    assert np.allclose(num, cam.jac(x), atol=1e-4), "Jacobian must match a numerical derivative"
    cam.yaw, cam.vf, cam.vl = np.array([0, 0.1]), np.array([10.0, 10.0]), np.zeros(2)
    y, _ = cam.ego(1, np.array([0, 10.0, 0, 0]), np.eye(4))
    assert y[0] > 0 and y[1] < 10, "a left turn while driving ahead moves a point ahead to the right and closer"

    # a walker crosses at constant speed, is hidden, and comes back under a new SAM id; a second walker appears far
    # away at the same moment and must NOT inherit the first walker's identity
    cam = Pixel(10, 1000, 500)

    def walker(hidden, relink=True):
        lay, ids = Layer(cam, relink), {}
        for k in range(50):
            dets = [] if 20 <= k < 20 + hidden else [(7 if k < 20 else 99, PED, np.array([100 + 8.0 * k, 200, 40, 100.0]))]
            if k >= 20 + hidden:
                dets.append((55, PED, np.array([800.0, 100, 40, 100])))
            for (sid, _, _), i in zip(dets, lay.step(k, dets)):
                ids.setdefault(sid, set()).add(i)
        return ids, lay
    ids, lay = walker(8)
    assert ids[7] == ids[99] and len(ids[7]) == 1, "hidden 0.8 s: the returning walker must get the old ID back"
    assert ids[55].isdisjoint(ids[7]), "the far walker must get a new ID"
    assert len(lay.relinks) == 1 and abs(lay.relinks[0][2] - 0.8) < 1e-9
    assert walker(15)[0][7].isdisjoint(walker(15)[0][99]), "hidden 1.5 s: longer than MEMORY, so the layer does not guess"
    assert walker(8, relink=False)[0][7].isdisjoint(walker(8, relink=False)[0][99]), "with relinking off, SAM's two IDs stay two IDs"
    lay = Layer(cam); box = np.array([100.0, 200, 40, 100])
    first = lay.step(0, [(7, PED, box)])
    for k in range(1, 200):
        lay.step(k, [])
    assert not lay.tracks and lay.step(200, [(7, PED, box)]) == first, "an ID SAM itself keeps is never changed, however long the gap"
    print("self-check passed")


if __name__ == "__main__":
    self_check()
    if len(sys.argv) > 3:
        bench, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
        flags = sys.argv[4:]
        for f in flags:                                            # NAME=value overrides a setting (used for the dev sweep)
            if "=" in f:
                globals()[f.split("=")[0]] = float(f.split("=")[1])
        (dst / "data").mkdir(parents=True, exist_ok=True)
        n = frames = secs = 0
        for f in sorted((src / "data").glob("*.txt")):
            a, b, c = run_file(bench, f, dst / "data" / f.name, pixel="--pixel" in flags, scratch="--scratch" in flags,
                               relink="--off" not in flags)
            n += a; frames += b; secs += c
        print(f"{dst.name}: {n} IDs given back after a gap, {frames} frames, {1000 * secs / max(frames, 1):.2f} ms per frame")
