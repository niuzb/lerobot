#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
exec 9>/tmp/vla_jepa_pipeline_20261011.lock
flock -n 9 || exit 90
rm -f vla_jepa_pipeline_20261011.exitcode
exec > vla_jepa_pipeline_20261011.log 2>&1
trap 'status=$?; printf "%s\n" "$status" > vla_jepa_pipeline_20261011.exitcode' EXIT
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
echo "$(date -Is) Waiting for pinned model assets"
for ((i=0; i<720; i++)); do
  if test -s outputs/vla_jepa_launch_20261011/prepared.json; then break; fi
  sleep 10
done
test -s outputs/vla_jepa_launch_20261011/prepared.json
"$UV" run --no-project --python "$PY" python vla_jepa_gpu_health_20261011.py
echo "$(date -Is) Starting checked preflight"
bash run_vla_jepa_checked_20261011.sh preflight 1 vla_jepa_tape_20261011_preflight_r3 > vla_jepa_tape_20261011_preflight_r3.log 2>&1
test -s outputs/vla_jepa_launch_20261011/vla_jepa_tape_20261011_preflight_r3.audit.rank0.json
test -s outputs/train/vla_jepa_tape_20261011_preflight_r3/checkpoints/000004/pretrained_model/model.safetensors
echo "$(date -Is) Preflight passed; starting formal training"
bash run_vla_jepa_checked_20261011.sh train 1 vla_jepa_tape_20261011_r1 > vla_jepa_tape_20261011_r1.log 2>&1
echo "$(date -Is) Formal training complete"
