#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
RUN=smolvla_tape_curated_20261002
PY=/root/miniconda3/envs/lerobot/bin/python
TRAIN=/root/miniconda3/envs/lerobot/bin/lerobot-train
DATA=/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1
test ! -e "outputs/train/${RUN}"
test ! -e "outputs/train/${RUN}_preflight"
trap 'status=$?; printf "%s\n" "$status" > "/root/code/lerobot/${RUN}.exitcode"' EXIT
"$PY" preflight_smolvla_curated_20261002.py
COMMON=(
  --policy.path=lerobot/smolvla_base
  '--rename_map={"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'
  --policy.device=cuda
  --dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1
  --dataset.root="$DATA"
  --dataset.video_backend=pyav
  --batch_size=8
  --num_workers=4
  --policy.scheduler_decay_steps=20000
  --policy.push_to_hub=false
  --wandb.enable=false
  --seed=1000
)
echo 'PHASE: two-step training and checkpoint preflight'
"$TRAIN" "${COMMON[@]}" --output_dir="outputs/train/${RUN}_preflight" --job_name="${RUN}_preflight" --steps=2 --save_freq=2 --log_freq=1
echo 'PHASE: fresh 20000-step training from smolvla_base'
"$TRAIN" "${COMMON[@]}" --output_dir="outputs/train/${RUN}" --job_name="$RUN" --steps=20000 --save_freq=5000 --log_freq=100 --policy.repo_id="zhniu/${RUN}"
