"""AV-stack side of the SAM 3 / SAM 3.1 perception service.

``Sam3Service`` launches ``sam3_server.py`` with the *SAM environment's* interpreter
(torch + Meta's ``sam3`` package) and talks JSON lines to it. The AV stack itself never
imports torch. Configure it with environment variables (or pass arguments):

    AVSTACK_SAM3_PYTHON       interpreter of the SAM env, e.g. .../Computervision_Research/.venv_sam3/bin/python
    AVSTACK_SAM3_CHECKPOINT   optional local sam3.pt for the per-frame image model (else HF cache)
    AVSTACK_SAM31_CHECKPOINT  optional local sam3.1_multiplex.pt for the video model (else HF cache)
    AVSTACK_SAM3_DEVICE       optional override: cuda | mps | cpu

``Sam3NativeBackend`` adapts the service to the open-vocab backend interface
(``segment(frame, prompts) -> list[Instance2D]``), so the rest of the stack (lifting,
fusion, tracking, planning) is unchanged whether instances come from the simulator's
emulator, SAM 3 on a Mac, or SAM 3.1 on an A100.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from avstack.common.types import CameraFrame
from avstack.config import OpenVocabConfig
from avstack.perception.open_vocab.backends import Instance2D

SERVER = Path(__file__).with_name("sam3_server.py")


class Sam3ServiceError(RuntimeError):
    pass


class Sam3Service:
    def __init__(self, python: str | None = None, checkpoint: str | None = None,
                 video_checkpoint: str | None = None, device: str | None = None, timeout: float = 900.0):
        python = python or os.environ.get("AVSTACK_SAM3_PYTHON")
        if not python or not Path(python).exists():
            raise Sam3ServiceError(
                "Set AVSTACK_SAM3_PYTHON to the SAM environment's python "
                "(the venv where Meta's `sam3` package and torch are installed).")
        env = dict(os.environ)
        for key, val in (("AVSTACK_SAM3_CHECKPOINT", checkpoint), ("AVSTACK_SAM31_CHECKPOINT", video_checkpoint),
                         ("AVSTACK_SAM3_DEVICE", device)):
            if val:
                env[key] = val
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.timeout = timeout
        self.proc = subprocess.Popen([python, "-u", str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, env=env, bufsize=1)
        self._lock = threading.Lock()
        info = self.request({"cmd": "ping"})
        self.device = info.get("device")

    def request(self, req: dict) -> dict:
        with self._lock:
            if self.proc.poll() is not None:
                raise Sam3ServiceError(f"SAM service exited with code {self.proc.returncode}")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            result: dict = {}

            def read():
                # skip anything that is not a protocol reply (defence against stray prints)
                while True:
                    line = self.proc.stdout.readline()
                    if not line or line.lstrip().startswith('{"ok"'):
                        result["line"] = line
                        return

            t = threading.Thread(target=read, daemon=True)
            t.start()
            t.join(self.timeout)
            if "line" not in result or not result["line"]:
                raise Sam3ServiceError(f"no reply from SAM service within {self.timeout:.0f}s for {req.get('cmd')}")
        reply = json.loads(result["line"])
        if not reply.get("ok"):
            raise Sam3ServiceError(reply.get("error", "unknown error"))
        return reply

    # -- conveniences ----------------------------------------------------- #
    def segment(self, image_path: str, prompts: list[str], threshold: float, mask_dir: str | None = None) -> dict:
        return self.request({"cmd": "segment", "image": str(image_path), "prompts": list(prompts),
                             "threshold": threshold, "mask_dir": mask_dir})

    def track_video(self, frames_dir: str, prompts: list[str], out_json: str, threshold: float = 0.4,
                    version: str = "sam3.1", max_objects: int = 16) -> dict:
        return self.request({"cmd": "track_video", "frames_dir": str(frames_dir), "prompts": list(prompts),
                             "out_json": str(out_json), "threshold": threshold, "version": version,
                             "max_objects": max_objects})

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request({"cmd": "shutdown"})
            except Sam3ServiceError:
                pass
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_mask(path: str, shape: list[int]) -> np.ndarray:
    """Masks travel as bit-packed .npy files (a 1242x375 mask is 58 kB, not 466 kB)."""
    bits = np.load(path)
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


class Sam3NativeBackend:
    """Open-vocab backend backed by Meta's native ``sam3`` package via the service.

    Needs real RGB frames: ``CameraFrame.image`` may be a file path, a PIL image or an
    HxWx3 uint8 array."""

    def __init__(self, cfg: OpenVocabConfig, service: Sam3Service | None = None):
        self.cfg = cfg
        self.service = service or Sam3Service()
        self._tmp = tempfile.TemporaryDirectory(prefix="avstack_sam3_")
        self.last_ms = 0.0

    def _as_path(self, frame: CameraFrame) -> str:
        img = frame.image
        if img is None:
            raise ValueError("Sam3NativeBackend needs real RGB frames (CameraFrame.image); "
                             "use backend='sim' with the simulator")
        if isinstance(img, (str, Path)):
            return str(img)
        from PIL import Image
        pil = img if hasattr(img, "save") else Image.fromarray(np.asarray(img, dtype=np.uint8))
        path = Path(self._tmp.name) / f"{frame.camera}.png"
        pil.save(path)
        return str(path)

    def segment(self, frame: CameraFrame, prompts: list[str]) -> list[Instance2D]:
        reply = self.service.segment(self._as_path(frame), prompts, self.cfg.score_threshold,
                                     mask_dir=str(Path(self._tmp.name) / "masks") if self.cfg.return_masks else None)
        self.last_ms = reply.get("ms", 0.0)
        out = []
        for inst in reply["instances"]:
            mask = load_mask(inst["mask"], inst["mask_shape"]) if inst.get("mask") else None
            out.append(Instance2D(frame.camera, inst["prompt"], inst["score"], tuple(inst["box"]), mask=mask))
        return out
