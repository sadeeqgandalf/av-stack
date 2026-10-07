#!/usr/bin/env bash
# (The official CLI cannot set OUTPUT_FOLDER: options defaulting to None are parsed as lists.
#  TrackEval's documented default writes results next to each tracker's data; we copy them
#  into results/ afterwards, leaving the official invocation untouched.)
# Official TrackEval evaluation (HOTA, CLEAR, Identity) of SAM 3, SAM 3.1 and the TrackR-CNN
# reference on KITTI-MOTS val and MOTS20 train. Default settings of the official scripts;
# only folder locations are passed. Run from /workspace/sam_mots_benchmark.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv_trackeval/bin/python
export MPLBACKEND=Agg                       # headless plotting (TrackEval writes plots at the end)
cd TrackEval

# KITTI-MOTS val
$PY "$ROOT/trackeval_run.py" scripts/run_kitti_mots.py --USE_PARALLEL False \
  --GT_FOLDER data/gt/kitti/kitti_mots_train --SPLIT_TO_EVAL val \
  --TRACKERS_FOLDER "$ROOT/trackers/kitti_mots" --TRACKERS_TO_EVAL SAM3 SAM3.1 > "$ROOT/logs/eval_kitti_mots.log" 2>&1
$PY "$ROOT/trackeval_run.py" scripts/run_kitti_mots.py --USE_PARALLEL False \
  --GT_FOLDER data/gt/kitti/kitti_mots_train --SPLIT_TO_EVAL val \
  --TRACKERS_FOLDER data/trackers/kitti/kitti_mots_val --TRACKERS_TO_EVAL track_rcnn > "$ROOT/logs/eval_kitti_mots_reference.log" 2>&1

# MOTS20 train: TrackEval expects <TRACKERS_FOLDER>/MOTS-train/<tracker>/data
mkdir -p "$ROOT/trackers/mots_layout"
ln -sfn "$ROOT/trackers/mots_challenge" "$ROOT/trackers/mots_layout/MOTS-train"
$PY "$ROOT/trackeval_run.py" scripts/run_mots_challenge.py --USE_PARALLEL False \
  --TRACKERS_FOLDER "$ROOT/trackers/mots_layout" --TRACKERS_TO_EVAL SAM3 SAM3.1 > "$ROOT/logs/eval_mots_challenge.log" 2>&1
$PY "$ROOT/trackeval_run.py" scripts/run_mots_challenge.py --USE_PARALLEL False \
  --TRACKERS_TO_EVAL track_rcnn > "$ROOT/logs/eval_mots_challenge_reference.log" 2>&1

collect() {   # $1 = tracker results folder, $2 = results/<benchmark>/<tracker>
  mkdir -p "$2"
  cp "$1"/*_summary.txt "$1"/*_detailed.csv "$2"/ 2>/dev/null || true
  cp "$1"/*_plot.png "$1"/*_plot.pdf "$2"/ 2>/dev/null || true
}
for t in SAM3 SAM3.1; do
  collect "$ROOT/trackers/kitti_mots/$t" "$ROOT/results/kitti_mots/$t"
  collect "$ROOT/trackers/mots_challenge/$t" "$ROOT/results/mots_challenge/$t"
done
collect data/trackers/kitti/kitti_mots_val/track_rcnn "$ROOT/results/kitti_mots/track_rcnn"
collect data/trackers/mot_challenge/MOTS-train/track_rcnn "$ROOT/results/mots_challenge/track_rcnn"

grep -h -E "^HOTA: |^COMBINED" "$ROOT"/logs/eval_*.log | awk 'NR%1==0' | grep -A1 "^HOTA" | grep -v "^--"
echo "results in $ROOT/results"
