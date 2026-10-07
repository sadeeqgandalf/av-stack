#!/usr/bin/env bash
# Stop all benchmark workers, clear queue state (finished outputs are kept), start new workers.
# usage: ./restart_workers.sh <n_sam3.1> <n_sam3>
cd "$(dirname "$0")"
pkill -f "^\.\./ComputerVision.*run_sam_mots"; pkill -f "^bash ./launch_workers"; sleep 6
stamp=$(date +%H%M); mkdir -p logs/restart_$stamp; mv logs/worker_*.log logs/restart_$stamp/ 2>/dev/null
rm -rf trackers/_claims; rm -f trackers/*/*/data/*.tmp
ls trackers/*/*/data/*.txt 2>/dev/null
setsid bash -c "./launch_workers.sh sam3.1 1 $1; ./launch_workers.sh sam3 $(( $1 + 1 )) $2" >/dev/null 2>&1 < /dev/null &
echo "started $1 x sam3.1, $2 x sam3"
