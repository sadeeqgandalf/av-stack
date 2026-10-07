#!/usr/bin/env bash
# Keep a CARLA server running: restart it within ~10 s if it dies (log: /tmp/carla_supervisor.log).
cd "$(dirname "$0")"
while true; do
  if ! ss -ltn | grep -q ':2000 '; then
    echo "$(date +%T) CARLA not listening: (re)starting" >> /tmp/carla_supervisor.log
    ./start_server.sh >> /tmp/carla_supervisor.log 2>&1
  fi
  sleep 10
done
