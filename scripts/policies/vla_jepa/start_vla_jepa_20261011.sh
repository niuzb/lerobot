#!/usr/bin/env bash
set -euo pipefail
cd /root/code/lerobot
exec bash scripts/policies/vla_jepa/train_vla_jepa_20261011.sh > vla_jepa_tape_20261011.log 2>&1
