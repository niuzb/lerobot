#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
nohup /root/miniconda3/envs/lerobot/bin/python -u smolvla_cloud_server.py > outputs/smolvla_cloud_server_20261003.log 2>&1 < /dev/null &
echo "SERVER_PID=$!"
