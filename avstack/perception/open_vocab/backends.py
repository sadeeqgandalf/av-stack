"""Promptable concept segmentation backends.

Every backend answers the same question SAM 3 answers: *given this image and
these noun-phrase prompts, return every matching instance* as (prompt, score,
box, mask). Downstream code never knows which backend ran.

  * ``Sam3Backend``        Meta's SAM 3 through Hugging Face transformers.
                           Needs real RGB frames (``CameraFrame.image``) and,
                           realistically, a GPU.
  * ``SimConceptBackend``  Emulates SAM 3 on the simulator's rendered frames,
                           including its failure modes: misses (worse for
                           small/occluded objects), score noise, box jitter and
                           hallucinated instances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avstack.common.types import CameraFrame
from avstack.config import OpenVocabConfig
from avstack.perception.open_vocab.ontology import ConceptMatcher


@dataclass
class Instance2D:
    camera: str
    prompt: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    description: str = ""          # free-text caption when the backend has one


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cross_prompt_nms(instances: list[Instance2D], iou: float = 0.6) -> list[Instance2D]:
    """One object can match several prompts ("person", "construction worker").
    Keep the highest-scoring (usually most specific) one per object."""
    keep: list[Instance2D] = []
    for inst in sorted(instances, key=lambda i: -i.score):
        if all(box_iou(inst.box_xyxy, k.box_xyxy) < iou for k in keep if k.camera == inst.camera):
            keep.append(inst)
    return keep


class SimConceptBackend:
    def __init__(self, cfg: OpenVocabConfig, rng: np.random.Generator, matcher: ConceptMatcher | None = None):
        self.cfg, self.rng = cfg, rng
        self.matcher = matcher or ConceptMatcher()

    def segment(self, frame: CameraFrame, prompts: list[str]) -> list[Instance2D]:
        c, rng = self.cfg, self.rng
        out: list[Instance2D] = []
        for obj in frame.objects:
            u0, v0, u1, v1 = obj.box_xyxy
            area_frac = (u1 - u0) * (v1 - v0) / (frame.width * frame.height)
            # small / occluded instances are missed more often, as with any real detector
            p_miss = c.sim_miss_prob + 0.25 * (1 - obj.visible_fraction) + 0.15 * float(area_frac < 2e-4)
            if rng.random() < p_miss:
                continue
            for prompt in prompts:
                sim = self.matcher.similarity(obj.description, prompt)
                if sim < 0.35:
                    continue
                score = float(np.clip(0.35 + 0.65 * sim * (0.6 + 0.4 * obj.visible_fraction)
                                      + rng.normal(0, 0.05), 0, 1))
                if score < c.score_threshold:
                    continue
                j = rng.normal(0, c.sim_box_px_std, 4)
                box = (u0 + j[0], v0 + j[1], u1 + j[2], v1 + j[3])
                out.append(Instance2D(frame.camera, prompt, score, box, description=obj.description))
        for _ in range(rng.poisson(c.sim_false_positive_rate)):
            u = rng.uniform(0, frame.width - 60)
            v = rng.uniform(frame.height * 0.5, frame.height * 0.7)
            w = rng.uniform(20, 120)
            out.append(Instance2D(frame.camera, str(rng.choice(prompts)), float(rng.uniform(0.45, 0.6)),
                                  (u, v, u + w, v + w * rng.uniform(0.5, 1.5)), description="hallucination"))
        return out


class Sam3Backend:
    """SAM 3 via transformers (verified against the HF ``Sam3Model`` docs).

        pip install torch transformers pillow
        huggingface-cli login     # facebook/sam3 is gated: accept the licence first
    """

    def __init__(self, cfg: OpenVocabConfig):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.cfg, self.torch = cfg, torch
        device_map = "auto" if cfg.device == "auto" else None
        self.model = Sam3Model.from_pretrained(cfg.sam3_model, device_map=device_map)
        if device_map is None:
            self.model = self.model.to(cfg.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(cfg.sam3_model)

    def segment(self, frame: CameraFrame, prompts: list[str]) -> list[Instance2D]:
        if frame.image is None:
            raise ValueError("Sam3Backend needs real RGB frames (CameraFrame.image); "
                             "use backend='sim' with the simulator")
        out: list[Instance2D] = []
        for prompt in prompts:
            inputs = self.processor(images=frame.image, text=prompt, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.cfg.score_threshold, mask_threshold=self.cfg.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            for mask, box, score in zip(res["masks"], res["boxes"], res["scores"]):
                out.append(Instance2D(frame.camera, prompt, float(score),
                                      tuple(float(x) for x in box.tolist()),
                                      mask=mask.cpu().numpy().astype(bool)))
        return out


def make_backend(cfg: OpenVocabConfig, rng: np.random.Generator, matcher: ConceptMatcher):
    if cfg.backend == "sam3_native":
        from avstack.perception.open_vocab.sam3_client import Sam3NativeBackend
        return Sam3NativeBackend(cfg)
    if cfg.backend == "sam3":
        return Sam3Backend(cfg)
    if cfg.backend == "sim":
        return SimConceptBackend(cfg, rng, matcher)
    raise ValueError(f"unknown open-vocab backend {cfg.backend!r}")
