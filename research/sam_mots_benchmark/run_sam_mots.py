#!/usr/bin/env python3
"""SAM 3 / SAM 3.1 zero-shot tracking on KITTI-MOTS (val) and MOTSChallenge (train),
written in the official MOTS tracker format for evaluation with TrackEval (HOTA).

Protocol (fixed before any result was seen; see PROTOCOL.md):
  * Models are built exactly as in Meta's official notebooks:
      SAM 3   -> build_sam3_video_predictor()                       (all defaults)
      SAM 3.1 -> build_sam3_multiplex_video_predictor(max_num_objects=128, use_fa3=False)
    max_num_objects=128 is the model class's own default ("should be able to cover nearly
    all cases"); the builder's 16 would cap SAM 3.1 below the densest frames of the
    benchmarks. use_fa3=False: FlashAttention 3 requires an H100 (hardware, not method).
  * Both models output at their default threshold (0.5). No other override.
  * Prompts are the benchmark class names ("car", "pedestrian"). One text prompt per
    pass, session reset between prompts, prompt on frame 0, propagate through the video
    (official flow).
  * The MOTS format requires non-overlapping masks: overlaps are resolved per pixel in
    favour of the higher-confidence object.
  * Output line: "<frame> <id> <class_id> <img_h> <img_w> <rle>" with class 1 = car,
    2 = pedestrian, id = class_id * 1000 + SAM object id. KITTI frames are 0-based,
    MOTSChallenge frames 1-based (each benchmark's own convention).

Usage (on the GPU host, in the SAM environment):
    python run_sam_mots.py --model sam3.1 --benchmark kitti_mots --out-root trackers
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch

DATA = Path("/workspace/ComputerVision_Research/Research_Data")
BENCHMARKS = {
    "kitti_mots": {
        "seqs": ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"],   # official val split
        "frames": lambda s: DATA / "data_tracking_image_2/training/image_02" / s,
        "out_name": lambda s: s,
        "frame_offset": 0,
        "classes": {"car": 1, "pedestrian": 2},
    },
    "mots_challenge": {
        "seqs": ["MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"],                       # official train split
        "frames": lambda s: DATA / "MOTSChallenge/train/images" / ("00" + s.split("-")[1]),
        "out_name": lambda s: s,
        "frame_offset": 1,
        "classes": {"pedestrian": 2},
    },
    # Test splits: labels are withheld, outputs are scored only by the benchmark servers.
    "kitti_mots_test": {
        "seqs": [f"{i:04d}" for i in range(29)],
        "frames": lambda s: DATA / "data_tracking_image_2/testing/image_02" / s,
        "out_name": lambda s: s,
        "frame_offset": 0,
        "classes": {"car": 1, "pedestrian": 2},
    },
    "mots_challenge_test": {
        "seqs": ["MOTS20-01", "MOTS20-06", "MOTS20-07", "MOTS20-12"],
        "frames": lambda s: DATA / "MOTSChallenge/MOTS/test" / s / "img1",
        "out_name": lambda s: s,
        "frame_offset": 1,
        "classes": {"pedestrian": 2},
    },
}


# Official start_session options that hold frames / inference state in host memory (exact copies).
# SAM 3.1's start_session ignores offload_state_to_cpu (its init_state does not take it).
SESSION_OPTIONS = {"offload_video_to_cpu": True, "offload_state_to_cpu": True}



def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# Setting variants for SAM 3.1 (not part of the scored protocol; see ekf/SETTINGS_EXPERIMENT.md). The official build
# is unchanged; these attributes are set on the built model afterwards. S = the values SAM 3 ships with
# (sam3/model_builder.py:756-772); M = the tracker's memory selection, which SAM 3 has on and SAM 3.1 off.
SAM3_TRACKING = {"score_threshold_detection": 0.5, "new_det_thresh": 0.7, "suppress_unmatched_only_within_hotstart": True,
                 "init_trk_keep_alive": 30, "max_trk_keep_alive": 30, "min_trk_keep_alive": -1,
                 "suppress_det_close_to_boundary": False, "fill_hole_area": 16, "masklet_confirmation_enable": False}
VARIANTS = {"S": SAM3_TRACKING, "M": {"use_memory_selection": True}, "SM": {**SAM3_TRACKING, "use_memory_selection": True}}


def apply_variant(pred, variant: str) -> dict:
    """Set each attribute on the top-level model if it has it, otherwise on every sub-module that does."""
    top = [v for v in vars(pred).values() if isinstance(v, torch.nn.Module)]
    done = {}
    for attr, new in VARIANTS[variant].items():
        holders = [m for m in top if hasattr(m, attr)] or [m for t in top for m in t.modules() if hasattr(m, attr)]
        assert holders, f"no module has {attr}"
        done[attr] = {"was": sorted({repr(getattr(m, attr)) for m in holders}), "now": new, "modules": [type(m).__name__ for m in holders]}
        for m in holders:
            setattr(m, attr, new)
    return done


def build(model: str, variant: str = ""):
    from sam3.model_builder import build_sam3_multiplex_video_predictor, build_sam3_video_predictor
    if model == "sam3":
        return build_sam3_video_predictor(), {"builder": "build_sam3_video_predictor", "kwargs": {}}
    kwargs = {"max_num_objects": 128, "use_fa3": False}
    pred = build_sam3_multiplex_video_predictor(**kwargs)
    # Official tracker option "to avoid GPU OOM on very long videos": per-frame outputs are kept in
    # host memory (exact copies; with use_memory_selection=False, as built, tracking logic is unchanged).
    mods = {id(m): m for v in vars(pred).values() if isinstance(v, torch.nn.Module) for m in v.modules()
            if hasattr(m, "offload_output_to_cpu_for_eval")}
    assert mods and not any(m.use_memory_selection for m in mods.values())
    for m in mods.values():
        m.offload_output_to_cpu_for_eval = True
    info = {"builder": "build_sam3_multiplex_video_predictor", "kwargs": kwargs,
            "post_build": {"tracker.offload_output_to_cpu_for_eval": True}}
    if variant:
        info["variant"] = {variant: apply_variant(pred, variant)}
        log(f"variant {variant}: {json.dumps(info['variant'])}")
    return pred, info


def offload_frame_cache(pred, session_id: str, fi: int):
    """Move the model's cached full-resolution masks of an already-yielded frame to host memory.
    SAM 3 / 3.1 keep `cached_frame_outputs` for every frame on the GPU (frames x objects x H x W), which
    exceeds 80 GB on 1080p MOTS20; the single-pass text-prompt flow never reads a yielded frame again.
    Exact copies; a later read would fail loudly on the device mismatch instead of changing results.
    Verified: KITTI-MOTS 0014 outputs byte-identical with and without (see PROTOCOL.md)."""
    cache = pred._all_inference_states[session_id]["state"].get("cached_frame_outputs")
    if cache is not None and fi in cache:
        cache[fi] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cache[fi].items()}


OFFLOAD_FRAME_CACHE = True


def run_prompt(pred, session_id: str, text: str) -> dict[int, list]:
    """Official flow: reset, text prompt on frame 0, propagate. -> {frame: [(obj_id, prob, rle)]}"""
    pred.handle_request({"type": "reset_session", "session_id": session_id})
    pred.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": text})
    per_frame: dict[int, list] = {}
    for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        fi = resp.get("frame_index")
        out = resp.get("outputs") or {}
        if fi is None:
            continue
        ids, probs, masks = out.get("out_obj_ids"), out.get("out_probs"), out.get("out_binary_masks")
        items = []
        if masks is not None and len(ids):
            masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
            ids = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            for k in range(len(ids)):
                m = masks[k].reshape(masks[k].shape[-2:]).astype(np.uint8)
                if m.any():
                    items.append((int(ids[k]), float(probs[k]), mask_util.encode(np.asfortranarray(m))))
        per_frame[int(fi)] = items
        if OFFLOAD_FRAME_CACHE:
            offload_frame_cache(pred, session_id, int(fi))
    return per_frame


def write_sequence(path: Path, per_class: dict[int, dict[int, list]], n_frames: int, frame_offset: int,
                   h: int, w: int) -> dict:
    """Merge classes, resolve overlaps by confidence, write official MOTS lines."""
    lines, stats = [], {"objects": set(), "overlap_pixels_resolved": 0, "dropped_after_overlap": 0}
    for fi in range(n_frames):
        cand = []
        for cls_id, frames in per_class.items():
            for obj_id, prob, rle in frames.get(fi, []):
                cand.append((prob, cls_id, obj_id, rle))
        cand.sort(key=lambda c: -c[0])
        taken = np.zeros((h, w), dtype=bool)
        for prob, cls_id, obj_id, rle in cand:
            m = mask_util.decode(rle).astype(bool)
            overlap = m & taken
            stats["overlap_pixels_resolved"] += int(overlap.sum())
            m &= ~taken
            if not m.any():
                stats["dropped_after_overlap"] += 1
                continue
            taken |= m
            enc = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            tid = cls_id * 1000 + obj_id
            stats["objects"].add(tid)
            lines.append(f"{fi + frame_offset} {tid} {cls_id} {h} {w} {enc['counts'].decode('ascii')}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    stats["objects"] = len(stats["objects"])
    stats["lines"] = len(lines)
    return stats


# Shared-GPU scheduling constants (GB), re-read before every booking so they can be tuned while running.
# Per-model working set of a loaded worker (booked for its lifetime) and memory growth per frame of a
# running sequence, fitted to measured peaks on this A100; MOTS20 (denser scenes) scaled by mots_factor.
SCHED_DEFAULTS = {"budget_gb": 76.0, "base_gb": {"sam3": 10.0, "sam3.1": 18.0},
                  "gb_per_frame": {"sam3": 0.035, "sam3.1": 0.040}, "mots_factor": 1.35}
SCHED_FILE = None                                    # set in main: <out-root>/_claims/scheduler.json


def sched() -> dict:
    try:
        return {**SCHED_DEFAULTS, **json.loads(SCHED_FILE.read_text())}
    except Exception:
        return SCHED_DEFAULTS


def est_growth_gb(model: str, benchmark: str, frames: int) -> float:
    c = sched()
    g = c["gb_per_frame"][model] * frames * (c["mots_factor"] if benchmark.startswith("mots_challenge") else 1.0)
    # cap: one sequence must always fit beside one loaded worker of each model
    return min(g, c["budget_gb"] - sum(c["base_gb"].values()))


class Ledger:
    """GPU memory booked by running sequences, shared by all workers (JSON file under flock).
    Entries of dead processes are dropped, so a killed worker never blocks the queue."""
    def __init__(self, path: Path):
        self.path, self.lock = path, path.with_suffix(".lock")

    def _edit(self, fn):
        with open(self.lock, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            book = json.loads(self.path.read_text()) if self.path.exists() else {}
            book = {k: v for k, v in book.items() if _alive(v["pid"])}
            ok = fn(book)
            self.path.write_text(json.dumps(book, indent=1))
            return ok

    def reserve(self, key: str, gb: float) -> bool:
        def fn(book):
            if sum(v["gb"] for v in book.values()) + gb > sched()["budget_gb"]:
                return False
            book[key] = {"gb": round(gb, 1), "pid": os.getpid()}
            return True
        return self._edit(fn)

    def keys(self) -> list[str]:
        return self._edit(lambda book: list(book))

    def release(self, key: str):
        self._edit(lambda book: book.pop(key, None))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_sequence(pred, bench, frame_dir, frames, out_file, h, w, tracker, worker, seq):
    sess = pred.handle_request({"type": "start_session", "resource_path": str(frame_dir),
                                **SESSION_OPTIONS})["session_id"]
    try:
        per_class = {}
        for text, cls_id in bench["classes"].items():
            tp = time.time()
            per_class[cls_id] = run_prompt(pred, sess, text)
            n_obj = len({o for items_ in per_class[cls_id].values() for o, _, _ in items_})
            log(f"{tracker} [{worker}] {seq} '{text}': {n_obj} objects, {time.time() - tp:.0f}s")
    finally:
        pred.handle_request({"type": "close_session", "session_id": sess})
    tmp = out_file.with_suffix(".tmp")
    stats = write_sequence(tmp, per_class, len(frames), bench["frame_offset"], h, w)
    tmp.rename(out_file)                               # appears atomically: renderers/evaluation never see partial files
    return stats


def main():
    """Queue worker. Several workers per model share the GPU: each takes the longest (benchmark, sequence)
    that is not done, not claimed (atomic mkdir) and whose estimated peak memory fits the shared GPU
    ledger; writes run_info_<worker>.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sam3", "sam3.1"], required=True)
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--variant", choices=list(VARIANTS), default="", help="SAM 3.1 only: changed tracking settings")
    a = ap.parse_args()
    assert not (a.variant and a.model == "sam3"), "variants change SAM 3.1 only"
    tracker = ("SAM3" if a.model == "sam3" else "SAM3.1") + (f"_{a.variant}" if a.variant else "")

    def frame_list(b, seq):
        return sorted(p for p in BENCHMARKS[b]["frames"](seq).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    items = [(b, seq, len(frame_list(b, seq))) for b in a.benchmark for seq in (a.seqs or BENCHMARKS[b]["seqs"])]
    items.sort(key=lambda it: it[2] * len(BENCHMARKS[it[0]]["classes"]), reverse=True)
    claims = a.out_root / "_claims"
    claims.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(claims / "_gpu_ledger.json")
    global SCHED_FILE
    SCHED_FILE = claims / "scheduler.json"

    base_key = f"{tracker}__{a.worker}__loaded"
    while not ledger.reserve(base_key, sched()["base_gb"][a.model]):
        time.sleep(30)
    t0 = time.time()
    pred, build_info = build(a.model, a.variant)
    load_s = time.time() - t0
    log(f"{a.model} [{a.worker}] loaded in {load_s:.0f}s")
    import sam3
    base = {
        "model": a.model, "tracker_name": tracker, "worker": a.worker, "build": build_info,
        "session_options": SESSION_OPTIONS, "frame_cache_offload": OFFLOAD_FRAME_CACHE, "load_seconds": load_s, "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "torch_cpu_threads": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "sam3_package": os.path.dirname(sam3.__file__), "hf_home": os.environ.get("HF_HOME"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    runs = {}

    def out_path(b, seq):
        return a.out_root / b / tracker / "data" / f"{BENCHMARKS[b]['out_name'](seq)}.txt"

    while True:
        todo = [it for it in items if not out_path(it[0], it[1]).exists()
                and not (claims / f"{tracker}__{it[0]}__{it[1]}").exists()]
        if not todo:
            break
        picked = None
        for b, seq, nf in todo:                        # longest first; a smaller one if the longest does not fit now
            key = f"{tracker}__{b}__{seq}"
            gb = est_growth_gb(a.model, b, nf)
            if not ledger.reserve(key, gb):
                continue
            try:
                (claims / key).mkdir()
            except FileExistsError:
                ledger.release(key)
                continue
            picked = (b, seq, key, gb)
            break
        if picked is None:
            # nothing fits now; surplus idle workers leave so their weights stop blocking big sequences
            mine = sorted(int(k.split("__w")[1].split("__")[0]) for k in ledger.keys()
                          if k.startswith(f"{tracker}__w") and k.endswith("__loaded"))
            if mine and int(a.worker[1:]) != mine[0]:
                log(f"{tracker} [{a.worker}]: nothing fits, leaving the rest to w{mine[0]}")
                break
            time.sleep(30)
            continue
        b, seq, key, gb = picked
        bench = BENCHMARKS[b]
        out_file = out_path(b, seq)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        run = runs.setdefault(b, dict(base, benchmark=b, prompts=bench["classes"], sequences={}))
        frames = frame_list(b, seq)
        from PIL import Image
        w, h = Image.open(frames[0]).size
        log(f"{tracker} [{a.worker}] {seq}: start ({len(frames)} frames, booked {gb:.0f} GB)")
        torch.cuda.reset_peak_memory_stats()
        ts, oom = time.time(), False
        try:
            stats = run_sequence(pred, bench, bench["frames"](seq), frames, out_file, h, w, tracker, a.worker, seq)
            stats.update(frames=len(frames), seconds=round(time.time() - ts, 1),
                         ms_per_frame=round(1000 * (time.time() - ts) / len(frames), 1), size=[w, h],
                         peak_gpu_gb=round(torch.cuda.max_memory_reserved() / 2**30, 1), booked_gpu_gb=round(gb, 1))
            run["sequences"][seq] = stats
            log(f"{tracker} [{a.worker}] {seq}: done {stats}")
        except torch.OutOfMemoryError:
            oom = True
        except Exception:
            run["sequences"][seq] = {"error": traceback.format_exc()}
            log(f"{tracker} [{a.worker}] {seq}: FAILED\n{traceback.format_exc()}")
        # (outside the except block, so the traceback no longer pins the session's tensors)
        gc.collect()
        torch.cuda.empty_cache()
        if oom:
            log(f"{tracker} [{a.worker}] {seq}: OOM (peak {torch.cuda.max_memory_reserved() / 2**30:.0f} GB, "
                f"booked {gb:.0f} GB), released for retry")
            (claims / key).rmdir()
        ledger.release(key)
        (a.out_root / b / tracker / f"run_info_{a.worker}.json").write_text(json.dumps(run, indent=1))
        if oom:
            time.sleep(60)
    for b, run in runs.items():
        run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (a.out_root / b / tracker / f"run_info_{a.worker}.json").write_text(json.dumps(run, indent=1))
    ledger.release(base_key)
    log(f"{tracker} [{a.worker}]: queue empty, all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
