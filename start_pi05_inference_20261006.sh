#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
test ! -e pi05_inference_20261006.pid
nohup /root/.local/bin/uv run --no-project --python /root/miniconda3/envs/lerobot/bin/python python -u pi05_cloud_server.py > pi05_inference_20261006.log 2>&1 < /dev/null &
echo "$!" > pi05_inference_20261006.pid
cat pi05_inference_20261006.pid
