"""Concept ontology: free-text phrases <-> canonical classes, and text similarity.

Open-vocabulary detectors (SAM 3, Grounding DINO, OWL-ViT) answer "where is
<phrase>?". The planner, however, needs to know *how a thing moves and how much
room to give it*. This module bridges the two:

  * ``canonical_class(phrase)`` maps "man in a hi-vis vest" -> PEDESTRIAN,
    "shredded truck tire" -> DEBRIS, "person riding a bike" -> BICYCLE,
    anything unrecognised -> UNKNOWN (treated conservatively downstream).
  * ``ConceptMatcher.similarity(a, b)`` scores how well a detection's phrase
    matches a prompt. The default embedder is lexical + ontology aware and runs
    anywhere; ``ClipTextEmbedder`` is a drop-in upgrade when a GPU is available.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from avstack.common.types import ObjectClass

CLASS_CONCEPTS: dict[ObjectClass, list[str]] = {
    ObjectClass.CAR: ["car", "sedan", "hatchback", "suv", "van", "minivan", "pickup", "pickup truck", "taxi", "coupe",
                      "vehicle", "police car", "wagon", "jeep"],
    ObjectClass.TRUCK: ["truck", "lorry", "semi", "trailer", "tanker", "box truck", "dump truck",
                        "tractor trailer", "18 wheeler"],
    ObjectClass.BUS: ["bus", "coach", "school bus", "tram", "shuttle"],
    ObjectClass.MOTORCYCLE: ["motorcycle", "motorbike", "scooter", "moped", "motorcyclist"],
    ObjectClass.BICYCLE: ["bicycle", "bike", "cyclist", "bicyclist", "e-bike"],
    ObjectClass.PEDESTRIAN: ["person", "pedestrian", "man", "woman", "child", "kid", "people", "worker",
                             "construction worker", "police officer", "officer", "jogger", "runner",
                             "flagger", "crossing guard", "stroller", "wheelchair"],
    ObjectClass.ANIMAL: ["animal", "deer", "dog", "cat", "horse", "cow", "moose", "bird", "fox", "sheep"],
    ObjectClass.TRAFFIC_CONE: ["traffic cone", "cone", "pylon", "delineator", "bollard"],
    ObjectClass.BARRIER: ["barrier", "jersey barrier", "barricade", "construction barrier", "fence",
                          "guardrail", "sawhorse", "road block"],
    ObjectClass.DEBRIS: ["debris", "tire", "tyre", "tread", "tire tread", "mattress", "box", "ladder", "rock",
                         "branch", "pallet", "trash", "bag", "cargo", "object on road", "pothole", "lumber"],
}

# words that end the head noun phrase ("tire ON the road", "man IN a vest")
_BREAKERS = {"on", "in", "with", "of", "at", "near", "by", "holding", "pushing", "carrying", "walking",
             "wearing", "crossing", "lying", "blocking", "from", "under", "behind", "next", "beside"}
_RIDE = {"riding", "rides", "ride", "on"}

_SYNONYM_TO_CLASS: dict[str, ObjectClass] = {}
for _cls, _words in CLASS_CONCEPTS.items():
    for _w in _words:
        _SYNONYM_TO_CLASS[_w] = _cls


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9\-]+", text.lower())
    out = []
    for t in toks:
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and t[:-1] in _SYNONYM_TO_CLASS:
            t = t[:-1]                      # plural -> singular for known concepts
        out.append(t)
    return out


def _find_class(tokens: list[str]) -> ObjectClass | None:
    """Last concept mentioned in the span (English head nouns come last), matching the
    longest phrase first so "pickup truck" is a car and never "pickup" + "truck"."""
    best, i = None, 0
    while i < len(tokens):
        for n in (3, 2, 1):
            phrase = " ".join(tokens[i:i + n])
            if phrase in _SYNONYM_TO_CLASS:
                best = _SYNONYM_TO_CLASS[phrase]
                i += n
                break
        else:
            i += 1
    return best


def canonical_class(phrase: str) -> ObjectClass:
    toks = _tokens(phrase)
    # "person riding a bicycle" -> the ridden thing decides the motion model
    for i, t in enumerate(toks):
        if t in ("riding", "rides", "ride"):
            ridden = _find_class(toks[i + 1:])
            if ridden in (ObjectClass.BICYCLE, ObjectClass.MOTORCYCLE, ObjectClass.ANIMAL):
                return ridden
    head = toks
    for i, t in enumerate(toks):
        if t in _BREAKERS and i > 0:
            head = toks[:i]
            break
    return _find_class(head) or _find_class(toks) or ObjectClass.UNKNOWN


# --------------------------------------------------------------------------- #
# Text embedders
# --------------------------------------------------------------------------- #
class LexicalEmbedder:
    """Deterministic, dependency-free phrase embedding.

    Features: word unigrams, the ontology class of the head noun (so "sedan" and
    "car" are close), and hashed character trigrams (robust to spelling variants).
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _h(self, key: str) -> int:
        return zlib.crc32(key.encode()) % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            toks = _tokens(text)
            for t in toks:
                if t in _BREAKERS or t in ("a", "an", "the"):
                    continue
                out[i, self._h("w:" + t)] += 1.0
                padded = f"#{t}#"
                for k in range(len(padded) - 2):
                    out[i, self._h("c:" + padded[k:k + 3])] += 0.15
            cls = canonical_class(text)
            if cls != ObjectClass.UNKNOWN:
                out[i, self._h("cls:" + cls.value)] += 2.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class ClipTextEmbedder:
    """CLIP text tower (GPU recommended). Same interface as LexicalEmbedder."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


class ConceptMatcher:
    def __init__(self, embedder=None):
        self.embedder = embedder or LexicalEmbedder()
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.embedder.encode([text])[0]
        return self._cache[text]

    def similarity(self, a: str, b: str) -> float:
        return float(self._vec(a) @ self._vec(b))

    def best_prompt(self, phrase: str, prompts: list[str]) -> tuple[int, float]:
        sims = [self.similarity(phrase, p) for p in prompts]
        i = int(np.argmax(sims))
        return i, sims[i]
