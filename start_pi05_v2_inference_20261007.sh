#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
test ! -e pi05_v2_inference_20261007.pid
nohup /root/.local/bin/uv run --no-project --python /root/miniconda3/envs/lerobot/bin/python python -u pi05_v2_cloud_server.py > pi05_v2_inference_20261007.log 2>&1 < /dev/null &
echo "$!" > pi05_v2_inference_20261007.pid
cat pi05_v2_inference_20261007.pid
