"""SAM 3 / SAM 3.1 perception service. Runs in the *SAM environment*, not the AV stack's.

Large vision models ship with heavy, GPU-specific dependencies (torch, CUDA/MPS, flash
attention). Production stacks keep them out of the driving process and run them as a
service; this is that service. It speaks JSON lines on stdin/stdout (logs go to stderr),
so it can run as a local subprocess today and on a GPU box tomorrow.

Launched by ``Sam3Service`` (sam3_client.py) with the SAM environment's interpreter, e.g.

    <sam3-venv>/bin/python -m avstack.perception.open_vocab.sam3_server   (with PYTHONPATH=AV-STACK)

Requests (one JSON object per line) and replies:

  {"cmd": "ping"}
      -> {"ok": true, "device": "mps", "torch": "2.x"}
  {"cmd": "segment", "image": "/path/frame.png", "prompts": ["car", "person"],
   "threshold": 0.45, "mask_dir": "/tmp/masks" | null}
      SAM 3 image model (per-frame promptable concept segmentation). The image encoder
      runs once; every prompt reuses its features.
      -> {"ok": true, "instances": [{"prompt", "score", "box": [x0,y0,x1,y1], "area",
                                     "mask": "/tmp/masks/xyz.npy" | null}], "ms": 812.4}
  {"cmd": "track_video", "frames_dir": "/path/jpgs", "prompts": [...], "threshold": 0.4,
   "version": "sam3.1", "max_objects": 16, "out_json": "/path/out.json"}
      SAM 3.1 Object Multiplex video predictor: masklets with persistent object ids.
      Writes per-frame instances to out_json (boxes, scores, ids, concept).
      -> {"ok": true, "out_json": "...", "frames": 40, "objects": 23, "ms": ...}
  {"cmd": "shutdown"}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def log(*a):
    print("[sam3-server]", *a, file=sys.stderr, flush=True)


class Sam3Server:
    def __init__(self):
        import torch
        from sam3.model.device_utils import get_device
        self.torch = torch
        self.device = os.environ.get("AVSTACK_SAM3_DEVICE") or get_device()
        self.image_checkpoint = os.environ.get("AVSTACK_SAM3_CHECKPOINT") or None
        self.video_checkpoint = os.environ.get("AVSTACK_SAM31_CHECKPOINT") or None
        self._processor = None
        self._predictors: dict = {}
        log(f"ready device={self.device} torch={torch.__version__}")

    # ------------------------------------------------------------------ image model
    def processor(self, threshold: float):
        if self._processor is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
            t0 = time.time()
            model = build_sam3_image_model(device=self.device, checkpoint_path=self.image_checkpoint,
                                           load_from_HF=self.image_checkpoint is None)
            self._processor = Sam3Processor(model, device=self.device, confidence_threshold=threshold)
            log(f"SAM 3 image model loaded in {time.time() - t0:.1f}s")
        self._processor.set_confidence_threshold(threshold)
        return self._processor

    def segment(self, req: dict) -> dict:
        import numpy as np
        from PIL import Image
        t0 = time.time()
        proc = self.processor(float(req.get("threshold", 0.45)))
        image = Image.open(req["image"]).convert("RGB")
        state = proc.set_image(image)                       # image encoder: once per frame
        mask_dir = req.get("mask_dir")
        if mask_dir:
            Path(mask_dir).mkdir(parents=True, exist_ok=True)
        out = []
        for prompt in req["prompts"]:
            res = proc.set_text_prompt(prompt=prompt, state=state)
            masks, boxes, scores = res["masks"], res["boxes"], res["scores"]
            for i in range(len(scores)):
                m = masks[i]
                m = m.detach().to("cpu").numpy() if hasattr(m, "detach") else np.asarray(m)
                m = m.reshape(m.shape[-2:]).astype(bool)
                path = None
                if mask_dir:
                    path = str(Path(mask_dir) / f"{uuid.uuid4().hex}.npy")
                    np.save(path, np.packbits(m), allow_pickle=False)
                b = boxes[i].detach().to("cpu").tolist() if hasattr(boxes[i], "detach") else list(boxes[i])
                s = float(scores[i].detach().to("cpu")) if hasattr(scores[i], "detach") else float(scores[i])
                out.append({"prompt": prompt, "score": s, "box": [float(v) for v in b],
                            "area": int(m.sum()), "mask": path, "mask_shape": list(m.shape)})
        return {"ok": True, "instances": out, "ms": (time.time() - t0) * 1e3}

    # ------------------------------------------------------------------ video (SAM 3.1)
    def _no_cuda_import_shim(self) -> None:
        """SAM 3.1's multiplex module queries ``torch.cuda.get_device_properties(0)`` at
        import time (to pick TF32 / fused-kernel settings), which raises on a CUDA-less torch.
        Without CUDA, report 'no modern GPU' so it takes its generic branch. Process-local:
        Meta's code and the research repo are not modified."""
        if self.torch.cuda.is_available():
            return
        from types import SimpleNamespace
        self.torch.cuda.get_device_properties = lambda *a, **k: SimpleNamespace(
            major=0, minor=0, name="none", total_memory=0, multi_processor_count=0)

    def predictor(self, version: str, max_objects: int):
        key = (version, max_objects)
        if key not in self._predictors:
            self._no_cuda_import_shim()
            from sam3.model.device_utils import install_cuda_compat_shim
            from sam3.model_builder import build_sam3_predictor
            dev = install_cuda_compat_shim()
            use_fa3 = str(dev).startswith("cuda") and self.torch.cuda.get_device_capability()[0] >= 9
            t0 = time.time()
            kwargs = dict(version=version, compile=False, async_loading_frames=False, warm_up=False,
                          use_fa3=use_fa3)
            if version == "sam3.1":
                kwargs["max_num_objects"] = max_objects
            if self.video_checkpoint and version == "sam3.1":
                kwargs["checkpoint_path"] = self.video_checkpoint
            p = build_sam3_predictor(**kwargs)
            for m in (p.model, getattr(p.model, "tracker", None)):
                if m is not None and hasattr(m, "fill_hole_area"):
                    m.fill_hole_area = 0
            self._predictors[key] = p
            log(f"{version} video predictor loaded in {time.time() - t0:.1f}s on {dev}")
        return self._predictors[key]

    def track_video(self, req: dict) -> dict:
        import numpy as np
        t0 = time.time()
        version = req.get("version", "sam3.1")
        pred = self.predictor(version, int(req.get("max_objects", 16)))
        thr = float(req.get("threshold", 0.4))
        sess = pred.handle_request({"type": "start_session", "resource_path": req["frames_dir"],
                                    "offload_video_to_cpu": True})["session_id"]
        frames: dict[int, list] = {}
        next_id, n_objects = 0, 0
        try:
            # SAM 3 video holds one text concept per pass: run each prompt, remap ids globally
            for concept in req["prompts"]:
                pred.handle_request({"type": "reset_session", "session_id": sess})
                pred.handle_request({"type": "add_prompt", "session_id": sess, "frame_index": 0,
                                     "text": concept, "output_prob_thresh": thr})
                local: dict[int, int] = {}
                for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sess,
                                                        "output_prob_thresh": thr}):
                    fi = resp.get("frame_index")
                    if fi is None:
                        continue
                    o = resp.get("outputs", {})
                    ids, masks = o.get("out_obj_ids", []), o.get("out_binary_masks")
                    probs = o.get("out_probs")
                    if masks is None:
                        continue
                    ids = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
                    masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
                    probs = (probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)) \
                        if probs is not None else np.ones(len(ids))
                    for k, oid in enumerate(ids):
                        m = masks[k].reshape(masks[k].shape[-2:]).astype(bool)
                        if not m.any():
                            continue
                        if int(oid) not in local:
                            local[int(oid)] = next_id
                            next_id += 1
                        ys, xs = np.nonzero(m)
                        frames.setdefault(int(fi), []).append({
                            "id": local[int(oid)], "prompt": concept, "score": float(probs[k]),
                            "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                            "area": int(m.sum())})
                n_objects += len(local)
                log(f"  {concept!r}: {len(local)} objects")
        finally:
            pred.handle_request({"type": "close_session", "session_id": sess})
        out = {"version": version, "prompts": req["prompts"], "threshold": thr,
               "frames": {str(k): v for k, v in sorted(frames.items())}}
        Path(req["out_json"]).write_text(json.dumps(out))
        return {"ok": True, "out_json": req["out_json"], "frames": len(frames), "objects": n_objects,
                "ms": (time.time() - t0) * 1e3}


def main():
    # stdout is the protocol channel. Libraries (sam3's logger, tqdm, torch) print there too,
    # so keep a private handle to the real stdout for replies and point fd 1 at stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply_line(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    server = Sam3Server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                reply = {"ok": True, "device": str(server.device), "torch": server.torch.__version__}
            elif cmd == "segment":
                reply = server.segment(req)
            elif cmd == "track_video":
                reply = server.track_video(req)
            elif cmd == "shutdown":
                reply_line({"ok": True})
                break
            else:
                reply = {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except Exception as e:                              # never die on a bad request
            log(traceback.format_exc())
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reply_line(reply)


if __name__ == "__main__":
    main()
