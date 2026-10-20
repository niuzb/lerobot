#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/code/lerobot/outputs/fastwam_20261020
cd "$ROOT"
exec >> "$ROOT/worker.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/worker.exitcode"' EXIT
export PYTHONPATH="$ROOT/code/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$ROOT/venv/bin/python"
CFG="$ROOT/fastwam_so101_20261020.json"
printf 'FASTWAM_WORKER_STARTED\n'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Downloads/preflight are CPU preparation; do not load an incomplete checkpoint.
for ((i=0; i<360; i++)); do
    if test -f "$ROOT/READY"; then break; fi
    if test -f "$ROOT/PREPARATION_FAILED"; then exit 3; fi
    sleep 60
done
test -f "$ROOT/READY"
"$PY" -u train_fastwam_so101_20261020.py --config "$CFG" --mode smoke
"$PY" -u train_fastwam_so101_20261020.py --config "$CFG" --mode validate \
    --checkpoint "$ROOT/smoke/checkpoints/000002/pretrained_model"
"$PY" -u train_fastwam_so101_20261020.py --config "$CFG" --mode train
"$PY" -u train_fastwam_so101_20261020.py --config "$CFG" --mode validate \
    --checkpoint /root/code/lerobot/outputs/train/fastwam_so101_mixed_20261020/checkpoints/020000/pretrained_model
printf 'FASTWAM_TRAINING_AND_RELOAD_COMPLETE\n'
