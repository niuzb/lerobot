#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
MODE=${1:?preflight or train}
GPUS=${2:?GPU count}
TAG=${3:?unique output tag}
RUN=vla_jepa_tape_20261011
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JEPA_AUDIT_PREFIX="/root/code/lerobot/outputs/vla_jepa_launch_20261011/${TAG}.audit"
trap 'status=$?; printf "%s\n" "$status" > "${TAG}.exitcode"' EXIT
test -s outputs/vla_jepa_launch_20261011/prepared.json
test ! -e "outputs/train/${TAG}"
COMMON=(
  --policy.path="/root/code/lerobot/outputs/${RUN}_base_assets"
  --policy.push_to_hub=false
  --dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1
  --dataset.root=/root/code/lerobot/datasets/pi05_tape_v2_20261006
  --dataset.video_backend=pyav --dataset.eval_split=0.1
  --num_workers=2 --env_eval_freq=0 --wandb.enable=false --seed=1000
  --accelerator.mixed_precision=bf16 --parallelism.dp_replicate="$GPUS"
  --output_dir="outputs/train/${TAG}" --job_name="$TAG"
)
if [ "$MODE" = preflight ]; then
  "$UV" run --no-project --python "$PY" python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    vla_jepa_preflight_checked_20261011.py "${COMMON[@]}" \
    --num_workers=0 --batch_size=2 --accelerator.gradient_accumulation.steps=2 \
    --steps=4 --eval_steps=4 --max_eval_samples=8 --save_freq=4 --log_freq=1
elif [ "$MODE" = train ]; then
  # Preserve 16 samples/update and 5,000 optimizer updates across GPU counts.
  test "$GPUS" -eq 1 -o "$GPUS" -eq 2 -o "$GPUS" -eq 4
  ACCUM=$((8 / GPUS))
  STEPS=$((40000 / GPUS))
  FREQ=$((5000 / GPUS))
  "$UV" run --no-project --python "$PY" python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    -m lerobot.scripts.lerobot_train "${COMMON[@]}" \
    --batch_size=2 --accelerator.gradient_accumulation.steps="$ACCUM" \
    --steps="$STEPS" --eval_steps="$FREQ" --max_eval_samples=0 --save_freq="$FREQ" --log_freq=40 \
    --policy.scheduler_warmup_steps=$((2000 / GPUS)) --policy.scheduler_decay_steps="$STEPS"
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
