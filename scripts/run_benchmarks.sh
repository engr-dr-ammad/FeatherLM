#!/usr/bin/env bash
# Reproduce all benchmark results in results/ (CPU). Takes ~1-2 hours on a 2-core machine.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
[ -f data/shakespeare/train.bin ] || python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192
# 1. Matched-budget quality comparison across architectures and seeds
python -m featherlm.benchmarks.benchmark_quality --spec configs/bench_micro.yaml --out results/quality
# 2. Peak training / inference memory (isolated processes)
python -m featherlm.benchmarks.benchmark_memory --configs configs/nano_cpu.yaml configs/nano_transformer_cpu.yaml \
  --steps 3 --contexts 256 1024 4096 --out results/memory
# 3. CPU inference speed vs context
python -m featherlm.benchmarks.benchmark_speed --configs configs/nano_cpu.yaml configs/nano_transformer_cpu.yaml \
  --contexts 128 512 2048 --out results/speed
# 4. Quantization + ONNX validation of the trained Nano model (if present)
if [ -f runs/featherlm-nano/final/config.json ]; then
  for m in fp16 int8 int4; do
    python -m featherlm.export.quantize --model runs/featherlm-nano/final --mode $m --out models/featherlm-nano-$m
  done
  python -m featherlm.export.onnx_export --model runs/featherlm-nano/final --out models/featherlm-nano-onnx
  python -m featherlm.export.validate --reference runs/featherlm-nano/final \
    --candidates models/featherlm-nano-fp16 models/featherlm-nano-int8 models/featherlm-nano-int4 \
    --onnx models/featherlm-nano-onnx --data data/shakespeare --out results/quantization.json
fi
