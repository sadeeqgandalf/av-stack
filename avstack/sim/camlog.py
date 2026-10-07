"""Camera logs: one on-disk format for simulator renders, CARLA recordings and real drives.

    <log>/
      meta.json                 format, source, scenario, dt, cameras (intrinsics + mounting), label table
      frames.jsonl              one line per frame: t, ego pose, ground-truth objects, stack tracks
      <camera>/rgb/00000.jpg    (SAM 3.1 reads this folder directly: zero-padded frame index)
      <camera>/depth/00000.png  uint16, planar depth in centimetres (0 = no return / sky)
      <camera>/semantic/00000.png  uint8 CARLA semantic tag
      <camera>/instance/00000.png  uint16, actor id + 1 (0 = none)
      <camera>/rgb.mp4, semantic.mp4, depth.mp4   control / preview videos (e.g. Cosmos-Transfer)

Anything that can fill these folders (our simulator, a CARLA bridge, a real vehicle log)
feeds the same downstream tools: SAM 3.1 auto-labelling, perception scoring, Cosmos.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from avstack.sim.render import CARLA_TAGS, CameraRenderer, colorize_semantic

FORMAT = "avstack-camlog/1"


class CamLogWriter:
    def __init__(self, out_dir: str | Path, scenario, cfg, cameras: list[str] | None = None,
                 width: int = 960, height: int = 540, every: int = 5, source: str = "avstack-sim", seed: int = 0):
        self.root = Path(out_dir)
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.scenario, self.cfg, self.every = scenario, cfg, every
        rig = {c.name: c for c in cfg.sensors.cameras}
        names = cameras or ["front"]
        self.renderers = {n: CameraRenderer(rig[n], scenario.map, width, height) for n in names}
        for n in names:
            for sub in ("rgb", "depth", "semantic", "instance"):
                (self.root / n / sub).mkdir(parents=True)
        self._frames = open(self.root / "frames.jsonl", "w")
        self._k, self.n = 0, 0
        meta = {
            "format": FORMAT, "source": source, "scenario": scenario.name, "description": scenario.description,
            "seed": seed, "dt": cfg.sim_dt * every,
            "cameras": [{"name": n, **r.intrinsics, "mount": {k: v for k, v in asdict(rig[n]).items()
                                                               if k in ("yaw", "mount_x", "mount_z", "hfov")}}
                        for n, r in self.renderers.items()],
            "semantic_tags": CARLA_TAGS, "depth_unit": "cm", "instance_offset": 1,
            "vocabulary": list(cfg.perception.open_vocab.vocabulary),
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=1))

    def on_step(self, world, stack) -> None:
        self._k += 1
        if self._k % self.every:
            return
        e = world.ego.state
        name = f"{self.n:05d}"
        for cam, r in self.renderers.items():
            out = r.render(e.x, e.y, e.yaw, world.actors)
            d = out["depth"]
            d_cm = np.where(np.isfinite(d), np.clip(d * 100.0, 0, 65535), 0).astype(np.uint16)
            base = self.root / cam
            Image.fromarray(out["rgb"]).save(base / "rgb" / f"{name}.jpg", quality=92)
            Image.fromarray(d_cm).save(base / "depth" / f"{name}.png")
            Image.fromarray(out["semantic"]).save(base / "semantic" / f"{name}.png")
            Image.fromarray(out["instance"]).save(base / "instance" / f"{name}.png")
        rec = {
            "i": self.n, "t": round(world.t, 3),
            "ego": {"x": e.x, "y": e.y, "yaw": e.yaw, "v": e.v},
            "objects": [{"id": a.id, "cls": a.cls.value, "description": a.description, "x": a.x, "y": a.y,
                         "yaw": a.yaw, "length": a.length, "width": a.width, "height": a.height, "v": a.v}
                        for a in world.actors],
            "tracks": [{"id": t.id, "cls": t.cls.value, "label": t.label, "x": t.x, "y": t.y, "yaw": t.yaw,
                        "length": t.length, "width": t.width, "speed": t.speed}
                       for t in stack.debug.tracks] if stack is not None else [],
        }
        self._frames.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self, videos: bool = True, fps: int | None = None) -> None:
        self._frames.close()
        if not videos or not shutil.which("ffmpeg"):
            return
        fps = fps or round(1.0 / (self.cfg.sim_dt * self.every))
        for cam in self.renderers:
            base = self.root / cam
            self._encode(base / "rgb" / "%05d.jpg", base / "rgb.mp4", fps)
            # colourised semantics and normalised depth as control/preview videos
            tmp = base / "_viz"
            tmp.mkdir(exist_ok=True)
            for p in sorted((base / "semantic").glob("*.png")):
                Image.fromarray(colorize_semantic(np.array(Image.open(p)))).save(tmp / f"s{p.stem}.png")
                d = np.array(Image.open(base / "depth" / p.name)).astype(np.float32) / 100.0
                d[d == 0] = 200.0
                dv = (255 * (1 - np.clip(np.log1p(d) / np.log1p(200.0), 0, 1))).astype(np.uint8)
                Image.fromarray(dv).save(tmp / f"d{p.stem}.png")
            self._encode(tmp / "s%05d.png", base / "semantic.mp4", fps)
            self._encode(tmp / "d%05d.png", base / "depth.mp4", fps)
            shutil.rmtree(tmp)

    @staticmethod
    def _encode(pattern: Path, out: Path, fps: int) -> None:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(pattern),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
