#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
exec 9>/tmp/vla_jepa_pipeline_20261011.lock
flock -n 9 || exit 73
TAG=vla_jepa_tape_20261011_r1_resume_20261012
test ! -e "${TAG}.log"
if pgrep -f '[l]erobot.scripts.lerobot_train' >/dev/null; then
  echo 'An existing trainer is running; refusing duplicate launch.' >&2
  exit 74
fi
exec >"${TAG}.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "${TAG}.exitcode"' EXIT
date -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/.local/bin/uv run --no-project --python /root/miniconda3/envs/lerobot/bin/python python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/root/code/lerobot/outputs/train/vla_jepa_tape_20261011_r1/checkpoints/015000/pretrained_model/train_config.json \
  --resume=true
