#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
RUN=smolvla_tape_into_pen_holder_20261001
trap 'status=$?; printf "%s\n" "$status" > "/root/code/lerobot/${RUN}.exitcode"' EXIT
/root/miniconda3/envs/lerobot/bin/lerobot-train \
  --config_path="/root/code/lerobot/outputs/train/${RUN}/checkpoints/000001/pretrained_model/train_config.json" \
  --resume=true \
  --steps=20000 \
  --save_freq=5000 \
  --policy.push_to_hub=true
