#!/usr/bin/env bash
# Render the four showcase videos from the exact tracker files TrackEval scores.
# Each render waits until its sequence's output file exists (written once per finished sequence).
set -uo pipefail
cd "$(dirname "$0")"
PY=/workspace/ComputerVision_Research/.venv_sam3/bin/python
D=/workspace/ComputerVision_Research/Research_Data
K=$D/data_tracking_image_2/training/image_02/0016
M=$D/MOTSChallenge/train/images/0009
render() {  # model tag bench seq frames offset fps scale title
  local f=trackers/$3/$2/data/$4.txt
  until [ -s "$f" ]; do sleep 60; done
  sleep 5
  $PY render_video.py --tracker-file "$f" --frames "$5" --frame-offset "$6" --fps "$7" --scale "$8" \
    --title "$9" --subtitle "zero-shot, text prompt, official TrackEval input file" --out "videos/${1}_${4}.mp4"
}
mkdir -p videos
render SAM3   SAM3   kitti_mots     0016      "$K" 0 10 1.5 "SAM 3 | KITTI-MOTS val 0016 | prompts: car, pedestrian" &
render SAM3.1 SAM3.1 kitti_mots     0016      "$K" 0 10 1.5 "SAM 3.1 | KITTI-MOTS val 0016 | prompts: car, pedestrian" &
render SAM3   SAM3   mots_challenge MOTS20-09 "$M" 1 30 1.0 "SAM 3 | MOTS20-09 (train) | prompt: pedestrian" &
render SAM3.1 SAM3.1 mots_challenge MOTS20-09 "$M" 1 30 1.0 "SAM 3.1 | MOTS20-09 (train) | prompt: pedestrian" &
wait
echo "ALL VIDEOS DONE"
