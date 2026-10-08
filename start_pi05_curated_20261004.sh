#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
RUN=pi05_tape_curated_20261004
if test -e "${RUN}.pid"; then
  test -e "${RUN}.exitcode" || { echo 'Previous run has no exit status; inspect it before restarting.'; exit 1; }
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  mv "${RUN}.log" "${RUN}.${STAMP}.log"
  mv "${RUN}.exitcode" "${RUN}.${STAMP}.exitcode"
  mv "${RUN}.pid" "${RUN}.${STAMP}.pid"
fi
nohup bash train_pi05_curated_20261004.sh > "${RUN}.log" 2>&1 < /dev/null &
echo "$!" > "${RUN}.pid"
cat "${RUN}.pid"
