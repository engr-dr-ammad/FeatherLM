#!/usr/bin/env bash
# Build the FeatherLM starter chatbot on CPU: pretrain Nano (if needed) -> starter conversations -> chat fine-tune.
# Developed by Ammad Aslam with AI-assisted tooling.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f runs/featherlm-nano/best/config.json ]; then
  [ -f data/shakespeare/train.bin ] || python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192
  echo "== Pretraining FeatherLM Nano (about 40 minutes on 2 CPU cores)"
  python train.py --config configs/nano_cpu.yaml
fi
echo "== Writing starter conversations"
python -m featherlm.data.chat_starter --out data/chat/starter
echo "== Fine-tuning into a chatbot (about 15 minutes on 2 CPU cores)"
python finetune.py --config configs/chat_nano.yaml --no-resume
python -m featherlm.benchmarks.chat_eval --model runs/featherlm-nano-chat/best --data data/chat/starter/eval_heldout.jsonl
echo "Done. Chat:  python inference.py --model runs/featherlm-nano-chat/best --chat --temperature 0.2"
echo "Or start Studio: python -m uvicorn api.main:app --port 8000  ->  http://127.0.0.1:8000"
