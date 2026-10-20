#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/code/lerobot/outputs/r2s2r_generation_20261018/runtime:/root/code/lerobot/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/root/miniconda3/envs/lerobot/bin/python
CFG=outputs/r2s2r_generation_20261018/scene_generation_merlin.json
PILOT=outputs/r2s2r_generation_20261018/pilot_physics_merlin_381
set +e
"$PY" -u -m lerobot.sim2real.experimental_generation --config "$CFG" \
  --output "$PILOT" --count 20 --seed 3000 --physics-only
code=$?
printf '%s\n' "$code" > r2s2r_pilot_381_20261018.exitcode
if test "$code" != 0; then exit "$code"; fi
"$PY" -u -m lerobot.sim2real.experiment_run --config "$CFG" \
  --root outputs/r2s2r_generation_20261018/workflow \
  --pilot "$PILOT" \
  --base /root/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --baseline outputs/train/r2s2r_baseline_20261018_t4 \
  --baseline-exitcode r2s2r_baseline_20261018_t4.exitcode \
  --mixed-output outputs/train/r2s2r_mixed_20261018
code=$?
printf '%s\n' "$code" > r2s2r_pipeline_20261018.exitcode
exit "$code"
