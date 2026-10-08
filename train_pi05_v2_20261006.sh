#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=pi05_tape_v2_20261006
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
MODEL=/root/code/lerobot/outputs/pi05_tape_curated_20261004_base_assets
TOKENIZER=/root/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c
trap 'status=$?; printf "%s\n" "$status" > "${RUN}.exitcode"' EXIT
test ! -e "outputs/train/${RUN}"
test ! -e "outputs/train/${RUN}_preflight"
"$UV" run --no-project --python "$PY" python prepare_pi05_v2_20261006.py
COMMON=(
  --policy.type=pi05
  --policy.pretrained_path="$MODEL"
  --policy.text_tokenizer_name="$TOKENIZER"
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.empty_cameras=1
  --policy.chunk_size=50
  --policy.n_action_steps=30
  --policy.use_relative_actions=false
  --dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1
  --dataset.root=/root/code/lerobot/datasets/pi05_tape_v2_20261006
  --dataset.video_backend=pyav
  --batch_size=4
  --num_workers=4
  --accelerator.gradient_accumulation.steps=4
  --policy.scheduler_decay_steps=40000
  --policy.scheduler_warmup_steps=2000
  --policy.push_to_hub=false
  --wandb.enable=false
  --dataset.eval_split=0.1
  --max_eval_samples=512
  --env_eval_freq=0
  --seed=1000
)
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" --output_dir="outputs/train/${RUN}_preflight" --job_name="${RUN}_preflight" --steps=4 --save_freq=4 --eval_steps=4 --max_eval_samples=8 --log_freq=1 2>&1 | tee "${RUN}.preflight.log"
grep -q 'Loaded state dict from model.safetensors' "${RUN}.preflight.log"
if grep -q 'Returning model without loading pretrained weights' "${RUN}.preflight.log"; then exit 1; fi
grep -q 'eval_loss=' "${RUN}.preflight.log"
test -s "outputs/train/${RUN}_preflight/checkpoints/000004/pretrained_model/model.safetensors"
echo 'PREFLIGHT_OK: starting fresh v2 from base; 40000 microbatches, held-out evaluation'
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" --output_dir="outputs/train/${RUN}" --job_name="$RUN" --steps=40000 --save_freq=5000 --eval_steps=2000 --log_freq=40 --policy.repo_id="zhniu/${RUN}"
