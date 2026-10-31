#!/usr/bin/env bash
# Prepare Tiny Shakespeare (public domain) and train FeatherLM Nano on CPU.
#   ./scripts/train.sh                       # Feather core
#   CONFIG=configs/nano_transformer_cpu.yaml ./scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/nano_cpu.yaml}
[ -f data/shakespeare/train.bin ] || python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192
python train.py --config "$CONFIG" "$@"
