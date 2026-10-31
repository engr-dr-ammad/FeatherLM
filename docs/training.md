# Training FeatherLM

*Developed by Ammad Aslam with AI-assisted tooling.*

Everything here runs on a CPU and no CUDA installation is needed. A GPU is used only when
you set `training.device` to `cuda` or `auto`.

## 1. Prepare data

Only use text you have the right to use. Nothing in this repository scrapes the web.

```bash
# Tiny Shakespeare: public-domain text, ~1.1 MB, ~300k tokens with an 8k vocabulary
python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192

# Your own licensed text files (one document per file)
python -m featherlm.data.prepare --input "corpus/*.txt" --out data/mycorpus --vocab-size 16384
python -m featherlm.data.prepare --input big.txt --split-paragraphs --out data/mycorpus

# Synthetic data for smoke tests (no download)
python -m featherlm.data.prepare --synthetic 2000 --out data/synthetic --vocab-size 512
```

Preparation cleans text (NFC, newlines, control characters), trains the byte-level BPE
tokenizer **on the training split only**, tokenizes, packs documents separated by EOS,
and writes `train.bin`, `val.bin` (uint16), `tokenizer.json` and `meta.json`.

## 2. Train from scratch

```bash
python train.py --config configs/nano_cpu.yaml
python train.py --config configs/nano_cpu.yaml --set training.max_steps=200 --set output_dir=runs/quick
python train.py --config configs/nano_transformer_cpu.yaml   # matched Transformer baseline
```

Outputs in `output_dir`:

| Path | Contents |
|---|---|
| `latest/` | resumable checkpoint (weights + optimizer + RNG state) |
| `best/` | best validation loss (weights only) |
| `final/` | weights at the end of the run (use this for inference) |
| `metrics.jsonl` | train/eval records: loss, lr, tokens/s, RSS, CPU% |
| `status.json` | live status (polled by the API / Studio) |
| `summary.json` | final numbers, including measured peak RSS |
| `run_config.json` | full config plus hardware and software versions |

**Interruptions.** Ctrl+C or SIGTERM finishes the current optimizer step, saves `latest/`
and exits. Re-running the same command resumes from `latest/` (pass `--no-resume` to
start over). Checkpoint directories are written atomically.

**Reproducibility.** Python, NumPy and PyTorch RNGs are seeded from `training.seed`; batch
sampling uses its own seeded generator, and RNG states are saved in checkpoints. Two runs
with the same seed on the same machine produce identical weights (tested).

## 3. Memory

Training refuses to start if its estimate exceeds the usable budget:

```
usable = min(training.memory_budget_mb, currently available RAM − training.reserve_mb)
```

It never sizes anything from total installed RAM. The estimate covers weights, gradients,
optimizer state, activations, logits and the measured runtime overhead. Actual peak RSS is
sampled every 50 ms and written to `summary.json`.

Measured on this repository's reference machine (2 vCPU Xeon, 7.8 GB RAM, PyTorch CPU):
FeatherLM Nano (13.2M params) with batch 8 × 256 tokens peaked at about **2.15 GB RSS**,
of which about 0.7 GB is the PyTorch runtime itself (the CUDA-enabled wheel; the CPU-only
wheel is smaller). See `docs/benchmarks.md`.

For 4 GB machines:

```bash
python train.py --config configs/nano_cpu.yaml --set training.batch_size=4 --set training.grad_accum=4 \
  --set model.gradient_checkpointing=true --set training.memory_budget_mb=2000
```

`training.optimizer: adafactor` reduces optimizer state from 2× to roughly 0.1× parameters.

## 4. Fine-tuning

`finetune.py` starts from an existing checkpoint (`finetune.init_from`) and evaluates on
the validation set before and after.

| Mode | Config | What it does |
|---|---|---|
| Continued pretraining | `data.format: bin` | more next-token training on new text |
| Supervised instruction tuning (full) | `data.format: jsonl`, `finetune.mode: full` | trains all weights on chat data; loss only on assistant tokens |
| LoRA adapters | `finetune.mode: lora` | freezes the model, trains rank-r adapters; saves the adapter and merged weights |

Set `finetune.set_chat_template: true` when training on chat data. Only checkpoints with
the `featherlm-v1` template are offered by `/v1/chat/completions`.

JSONL format, one example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "completion": "..."}
```

Example config:

```yaml
model: {}
data: {path: data/mysft, format: jsonl, train_file: data/mysft/train.jsonl, val_file: data/mysft/val.jsonl, seq_len: 256}
training: {batch_size: 8, max_steps: 500, lr: 3.0e-4, warmup_steps: 20, eval_interval: 100, save_interval: 100}
finetune: {init_from: runs/featherlm-nano/final, mode: lora, lora_r: 8, lora_alpha: 16, set_chat_template: true}
output_dir: runs/featherlm-nano-sft
```

A model trained on Tiny Shakespeare will not follow instructions; fine-tuning a model this
small on a small instruction set teaches format, not capability.

## 5. Cloud training

Any Ubuntu VM works; nothing requires a paid service.

```bash
# CPU VM (e.g. 8 vCPU / 32 GB)
git clone <repo> featherlm && cd featherlm && ./scripts/setup.sh --no-frontend
python -m featherlm.data.prepare --input "/mnt/corpus/*.txt" --out data/mycorpus --vocab-size 16384
nohup python train.py --config configs/tiny_cpu.yaml > train.log 2>&1 &

# Put output_dir on a persistent disk so preemption does not lose checkpoints:
python train.py --config configs/tiny_cpu.yaml --set output_dir=/mnt/persistent/runs/tiny
# After an interruption, the same command resumes from /mnt/persistent/runs/tiny/latest.
```

GPU instance: install a CUDA build of PyTorch and add `--set training.device=cuda` (or
`auto`). `--set training.precision=bf16` enables bf16 autocast on GPUs that support it. The
CUDA path has not been tested in this repository's CI environment, which has no GPU.

Containerized training:

```bash
docker build -t featherlm:cpu .
docker run --rm -v $PWD/data:/app/data -v $PWD/runs:/app/runs featherlm:cpu \
  python train.py --config configs/nano_cpu.yaml
```

## 6. Training from the API / Studio

`POST /training/jobs` starts `train.py` as a background process with server-enforced
limits (max steps, memory budget, one job at a time by default). Stop sends SIGTERM (a
checkpoint is saved); resume restarts from `latest/`. These endpoints require
`FEATHERLM_ADMIN_KEY`, or a direct loopback connection when no key is set.
