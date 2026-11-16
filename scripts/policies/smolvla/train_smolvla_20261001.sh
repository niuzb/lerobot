#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
RUN=smolvla_tape_into_pen_holder_20261001
trap 'status=$?; printf "%s\n" "$status" > "/root/code/lerobot/${RUN}.exitcode"' EXIT
/root/miniconda3/envs/lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --policy.device=cuda \
  --dataset.repo_id=zhniu/lerobot_zihao_dataset_a_20260927_202323 \
  --dataset.revision=01241e32b3b32f043ffb786a4d9afb264b4f981a \
  --output_dir="outputs/train/${RUN}" \
  --job_name="$RUN" \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --policy.scheduler_decay_steps=20000 \
  --wandb.enable=false \
  --policy.repo_id="zhniu/${RUN}" \
  --policy.push_to_hub=true
