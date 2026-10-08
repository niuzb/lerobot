#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
RUN=pi05_tape_curated_20261004
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
test ! -e "outputs/train/${RUN}"
trap 'status=$?; printf "%s\n" "$status" > "/root/code/lerobot/${RUN}.formal.exitcode"' EXIT
"$UV" run --no-project --python "$PY" python verify_pi05_preflight_20261004.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
TOKENIZER=$("$UV" run --no-project --python "$PY" python -c 'import json; print(json.load(open("pi05_tape_curated_20261004.assets.json"))["tokenizer_path"])')
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --policy.pretrained_path=/root/code/lerobot/outputs/pi05_tape_curated_20261004_base_assets \
  --policy.text_tokenizer_name="$TOKENIZER" \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.empty_cameras=1 --policy.chunk_size=50 --policy.n_action_steps=30 \
  --policy.use_relative_actions=false \
  --dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1 \
  --dataset.root=/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1 \
  --dataset.video_backend=pyav \
  --batch_size=4 --num_workers=4 --accelerator.gradient_accumulation.steps=4 \
  --policy.scheduler_decay_steps=80000 --policy.scheduler_warmup_steps=4000 \
  --policy.push_to_hub=false --wandb.enable=false --seed=1000 \
  --output_dir="outputs/train/${RUN}" --job_name="$RUN" \
  --steps=80000 --save_freq=20000 --log_freq=40 --policy.repo_id="zhniu/${RUN}"
