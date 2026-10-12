#!/usr/bin/env bash
set -euo pipefail
nohup bash /root/code/lerobot/start_vla_jepa_pipeline_20261011.sh </dev/null >/root/code/lerobot/vla_jepa_pipeline_20261011.dispatch.log 2>&1 &
echo "Pipeline PID $!"
