#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
RUN=smolvla_tape_curated_20261002
test ! -e "${RUN}.log"
test ! -e "outputs/train/${RUN}"
test ! -e "outputs/train/${RUN}_preflight"
nvidia-smi -L
nohup bash train_smolvla_curated_20261002.sh > "${RUN}.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "${RUN}.pid"
printf 'STARTED_PID=%s\n' "$pid"
