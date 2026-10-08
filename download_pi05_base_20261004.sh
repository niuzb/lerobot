#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1
trap 'status=$?; printf "%s\n" "$status" > pi05_base_20261004.download.exitcode' EXIT
/root/.local/bin/uv run --no-project --python /root/miniconda3/envs/lerobot/bin/python python download_pi05_base_20261004.py
