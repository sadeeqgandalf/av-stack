#!/usr/bin/env bash
# Start N queue workers per model: ./launch_workers.sh <model> <first_worker_index> <count>
cd "$(dirname "$0")"
export HF_HOME=/workspace/.cache/huggingface OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq "$2" $(( $2 + $3 - 1 ))); do
  nohup ../ComputerVision_Research/.venv_sam3/bin/python run_sam_mots.py --model "$1" \
    --benchmark kitti_mots mots_challenge --out-root trackers --worker "w$i" \
    > "logs/worker_${1}_w$i.log" 2>&1 < /dev/null &
  sleep 20
done
