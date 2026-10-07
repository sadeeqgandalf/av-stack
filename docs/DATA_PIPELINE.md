# Data pipeline: camera logs → SAM 3.1 → scoring → Cosmos

```
our simulator ─┐                          ┌─► SAM 3.1 (CUDA GPU) ─► masklets ──┐
CARLA (bridge) ┼─► camera log  ───────────┼─► Cosmos-Transfer (seg + depth) ─► photoreal video ─► SAM 3.1
real drives ───┘   (avstack-camlog/1)     └─► score_perception.py ◄───────────────────────────────────┘
```

## 1. Write a camera log (laptop, no GPU)

```bash
PYTHONPATH=. .venv/bin/python scripts/export_camlog.py crosswalk_multiple_threat --cameras front front_tele
# -> logs/crosswalk_multiple_threat_seed0/{meta.json, frames.jsonl, front/, front_tele/}
```

Per camera: `rgb/00000.jpg …`, `depth/*.png` (uint16 cm), `semantic/*.png` (CARLA tags),
`instance/*.png` (actor id + 1), plus `rgb.mp4`, `semantic.mp4`, `depth.mp4`.
`frames.jsonl` holds ego pose, ground-truth objects and the stack's own tracks per frame.

## 2. Score the stack's perception (laptop)

```bash
PYTHONPATH=. .venv/bin/python scripts/score_perception.py logs/crosswalk_multiple_threat_seed0 --camera front --pred tracks
```

## 3. SAM 3.1 on the GPU pod

SAM 3.1's Object Multiplex video model needs CUDA (on a Mac it fails at `torch.cuda` calls).
On the pod, reuse the environment where Meta's `sam3` package is installed; it also has
what the AV-stack client needs (numpy).

```bash
# from the laptop: send code + logs (adjust host / paths)
rsync -avz --exclude .venv --exclude viewer --exclude videos ./ POD:/workspace/AV-STACK/

# on the pod
cd /workspace/AV-STACK
export AVSTACK_SAM3_PYTHON=$(which python)          # the sam3 env's interpreter
PYTHONPATH=. python scripts/sam31_label.py logs/crosswalk_multiple_threat_seed0 --cameras front front_tele

# back on the laptop: fetch masklets and score SAM 3.1 next to the stack
rsync -avz POD:/workspace/AV-STACK/logs/ ./logs/ --include '*/' --include '*.json' --exclude '*'
PYTHONPATH=. .venv/bin/python scripts/score_perception.py logs/crosswalk_multiple_threat_seed0 --pred tracks sam31
```

Quick local check without CUDA (per-frame SAM 3 image model, slow, detection metrics only):
`scripts/sam31_label.py <log> --mode image --max-frames 8`, then `--pred sam3`.

## What to run SAM 3.1 on (and what not)

Our simulator's RGB is a flat-shaded proxy (people are boxes). **SAM 3 finds nothing in it**
(0 detections on 8 rendered frames, vs 39 instances on one real KITTI frame), so labelling
our renders is pointless. Run SAM 3.1 on:

1. **Real drives** (KITTI / BDD / your own logs): auto-labels and long-tail search.
2. **CARLA recordings**: photoreal pixels with exact ground truth, so SAM 3.1 *and* our
   perception get scored against truth.
3. **Cosmos-Transfer outputs**: generated from our (or CARLA's) `semantic.mp4` + `depth.mp4`
   control videos; SAM 3.1 then checks the generated video still contains what the scenario
   placed there, and gives the stack realistic camera input.

## Cosmos

The control videos are `<log>/<camera>/semantic.mp4` (CARLA/Cityscapes colours) and
`depth.mp4` (log-scaled, near = bright). Check Cosmos-Transfer's current input spec
(resolution, frame count, colour conventions) before a run; re-export with
`--size` to match (e.g. `--size 1280x704`).
