#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
test ! -e pi05_tape_v2_20261006.pid
nohup bash train_pi05_v2_20261006.sh > pi05_tape_v2_20261006.log 2>&1 < /dev/null &
echo $! > pi05_tape_v2_20261006.pid
cat pi05_tape_v2_20261006.pid
