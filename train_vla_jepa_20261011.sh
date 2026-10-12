#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
UV=/root/.local/bin/uv
PY=/root/miniconda3/envs/lerobot/bin/python
RUN=vla_jepa_tape_20261011
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY"
trap 'status=$?; printf "%s\n" "$status" > "${RUN}.exitcode"' EXIT
test ! -e "outputs/train/${RUN}"
test ! -e "outputs/train/${RUN}_preflight"
"$UV" run --no-project --python "$PY" python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'
"$UV" run --no-project --python "$PY" python -c 'import diffusers, qwen_vl_utils; from transformers import Qwen3VLForConditionalGeneration, AutoVideoProcessor; print("DEPENDENCIES_OK")'
"$UV" run --no-project --python "$PY" python prepare_vla_jepa_20261011.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COMMON=(
  --policy.path="/root/code/lerobot/outputs/${RUN}_base_assets"
  --policy.push_to_hub=false
  --dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1
  --dataset.root=/root/code/lerobot/datasets/pi05_tape_v2_20261006
  --dataset.video_backend=pyav
  --dataset.eval_split=0.1
  --num_workers=2
  --env_eval_freq=0
  --wandb.enable=false
  --seed=1000
)
# Full JEPA branch must pass forward/backward, held-out loss, and save before formal launch.
"$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" \
  --batch_size=2 --accelerator.gradient_accumulation.steps=2 \
  --output_dir="outputs/train/${RUN}_preflight" --job_name="${RUN}_preflight" \
  --steps=2 --eval_steps=2 --max_eval_samples=4 --save_freq=2 --log_freq=1 \
  2>&1 | tee "${RUN}.preflight.log"
grep -q 'eval_loss=' "${RUN}.preflight.log"
test -s "outputs/train/${RUN}_preflight/checkpoints/000002/pretrained_model/model.safetensors"
echo 'Preflight completed. Review finite action_loss/wm_loss, gradient flow, memory, and six-dimensional predictions before formal training.'
# Deliberately require inspection of the new architecture's preflight before long training.
# Formal command (run after that inspection):
# "$UV" run --no-project --python "$PY" python -m lerobot.scripts.lerobot_train "${COMMON[@]}" \
#   --batch_size=2 --accelerator.gradient_accumulation.steps=8 \
#   --output_dir="outputs/train/${RUN}" --job_name="$RUN" \
#   --steps=40000 --eval_steps=5000 --max_eval_samples=0 --save_freq=5000 --log_freq=40
