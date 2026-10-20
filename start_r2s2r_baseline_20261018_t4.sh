#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/code/lerobot
RUN=outputs/train/r2s2r_baseline_20261018_t4
test ! -e "$RUN"
PY=/root/miniconda3/envs/lerobot/bin/python
"$PY" -c 'import torch; x=torch.randn(512,512,device="cuda"); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(torch.__version__,torch.cuda.get_device_name(0))'
set +e
"$PY" -u so101_r2s2r.py train \
  --config configs/so101_r2s2r.json \
  --base /root/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --variant baseline --device cuda --output "$RUN"
code=$?
printf '%s\n' "$code" > r2s2r_baseline_20261018_t4.exitcode
exit "$code"
