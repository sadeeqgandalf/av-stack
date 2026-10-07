#!/usr/bin/env bash
# Start CARLA 0.9.14 headless on the GPU (no X server needed with -RenderOffScreen).
set -u
CARLA=/lambda/nfs/Simulation/carla
pkill -f CarlaUE4-Linux-Shipping 2>/dev/null; sleep 2
cd "$CARLA" && nohup ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000 \
  > /tmp/carla_server.log 2>&1 < /dev/null &
for _ in $(seq 1 90); do ss -ltn | grep -q ':2000 ' && { echo "CARLA up on :2000"; exit 0; }; sleep 2; done
echo "CARLA failed to start"; tail -40 /tmp/carla_server.log; exit 1
