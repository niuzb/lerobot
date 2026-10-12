#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
trap 'status=$?; printf "%s\n" "$status" > vla_jepa_assets_20261011.exitcode' EXIT
/root/.local/bin/uv run --no-project --python /root/miniconda3/envs/lerobot/bin/python python -u prepare_vla_jepa_20261011.py
