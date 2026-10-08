#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
RUN=pi05_tape_curated_20261004
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
test ! -e "outputs/train/${RUN}"
PREFLIGHT=${RUN}_preflight_v3
test ! -e "outputs/train/${PREFLIGHT}"
trap 'status=$?; printf "%s\n" "$status" > "/root/code/lerobot/${RUN}.exitcode"' EXIT
"$UV" run --no-project --python "$PY" python prepare_pi05_curated_20261004.py
MODEL=$("$UV" run --no-project --python "$PY" python -c 'import json; print(json.load(open("pi05_tape_curated_20261004.assets.json"))["model_path"])')
TOKENIZER=$("$UV" run --no-project --python "$PY" python -c 'import json; print(json.load(open("pi05_tape_curated_20261004.assets.json"))["tokenizer_path"])')
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
  --dataset.root=/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1
  --dataset.video_backend=pyav
  --batch_size=4
  --num_workers=4
  --accelerator.gradient_accumulation.steps=4
  --policy.scheduler_decay_steps=80000
  --policy.scheduler_warmup_steps=4000
  --policy.push_to_hub=false
  --wandb.enable=false
  --seed=1000
)
echo 'PHASE: four-microbatch Pi05 training, optimizer update and checkpoint preflight'
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" --output_dir="outputs/train/${PREFLIGHT}" --job_name="${PREFLIGHT}" --steps=4 --save_freq=4 --log_freq=1 2>&1 | tee "${RUN}.preflight.log"
# The current Pi05 loader can return random weights on a download error; require the success marker.
grep -q 'Loaded state dict from model.safetensors' "${RUN}.preflight.log"
if grep -q 'Returning model without loading pretrained weights' "${RUN}.preflight.log"; then exit 1; fi
test -s "outputs/train/${PREFLIGHT}/checkpoints/000004/pretrained_model/model.safetensors"
"$UV" run --no-project --python "$PY" python verify_pi05_preflight_20261004.py
echo 'PHASE: fresh 80000 microbatches / 20000 optimizer updates from pinned base'
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" --output_dir="outputs/train/${RUN}" --job_name="$RUN" --steps=80000 --save_freq=20000 --log_freq=40 --policy.repo_id="zhniu/${RUN}"
