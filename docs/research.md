# FeatherLM research report

*Developed by Ammad Aslam with AI-assisted tooling.*

**Status: preliminary.** Experiments are small (4M-13M parameters, one 300k-token corpus,
CPU budgets of minutes to an hour). They test whether the pipeline and the hypotheses are
workable; they do not establish that the Feather architecture is better than a Transformer.

## 1. Motivation

Training and running language models is usually gated on accelerator memory. On CPUs and
phones, the limits are RAM, memory bandwidth and energy. Two quantities matter most:

1. **Training memory**: weights, gradients, optimizer state and activations.
2. **Inference state**: a Transformer's KV cache grows linearly with context, which bounds
   context length on small devices.

FeatherLM asks whether a compact hybrid of gated linear recurrence and sparse local
attention can deliver similar quality per parameter as a Transformer while keeping
inference state constant and training feasible on an ordinary CPU, measured as
**quality per unit of memory, compute, latency and energy**, not parameter count alone.

## 2. Related work

* **Efficient Transformers:** GQA (Ainslie et al., 2023) and MQA shrink the KV cache;
  sliding-window attention (Longformer, Mistral) bounds it.
* **Linear recurrences and state-space models:** S4 (Gu et al., 2021), Mamba (Gu & Dao,
  2023) with input-dependent selection, RWKV (Peng et al., 2023), RetNet (Sun et al., 2023),
  and the RG-LRU of Griffin/Hawk (De et al., 2024).
* **Hybrids:** Griffin interleaves RG-LRU blocks with local attention; Jamba and others mix
  Mamba with attention. FeatherLM's hybrid is closest to Griffin and borrows its recurrence.
* **Parameter sharing:** Universal Transformers and ALBERT share layers across depth.
* **Compression:** weight-only post-training quantization (e.g. LLM.int8, GPTQ, AWQ)
  and LoRA for cheap adaptation.

## 3. Proposed architecture

Full details and equations are in `docs/architecture.md`. In short:

* Decoder-only; tied (optionally factorized) embeddings; pre-norm RMSNorm; SwiGLU FFNs.
* Most layers mix tokens with an RG-LRU-style gated recurrence preceded by a width-4 causal
  depthwise convolution; every third layer uses sliding-window GQA attention (window 64).
* Recurrence gates use a low-rank projection (d → d/4 → 2d).
* A chunked log-space scan computes the recurrence with O(T·d) activation memory.
* Optional depth-wise sharing (`n_unique_blocks`) trades parameters for repeated compute.

**Established vs. new.** The recurrence, convolution, hybrid layout, local attention, GQA,
RoPE, RMSNorm, SwiGLU, tying, factorization and sharing are all prior work. The candidate
contributions of this project are narrow: (a) low-rank recurrence gates, (b) the bounded-decay
chunked scan as a simple, dependency-free CPU implementation, and (c) the empirical study of
these choices under explicit CPU memory budgets. None of these has been shown to be novel
in the literature survey sense; a proper search is required before claiming novelty.

## 4. Complexity

| Quantity | Transformer | Feather (hybrid) |
|---|---|---|
| Mixer params / layer | 2d² + 2d·d_kv | ≈ 3.75d² (recurrent), 2d² + 2d·d_kv (local attn) |
| Training compute / token | O(L(d² + df) + L T d) | O(L(d² + df) + L_attn w d) |
| Decode compute / token | O(L(d² + T d)) | O(L d²) |
| Inference state | 2 L T d_kv | L_rec·K·d + 2 L_attn (w−1) d_kv (constant in T) |
| Training activations | O(L T d) | O(L T d), larger constant from the scan |

## 5. Experimental setup

* **Data:** Tiny Shakespeare (public domain), cleaned; byte-level BPE with 8,192 tokens
  trained on the training split; 300,695 training and 17,794 validation tokens (last ~5% of
  the text held out).
* **Hardware:** 2-vCPU Intel Xeon VM, 7.8 GB RAM, no GPU; PyTorch 2.15 (CPU execution).
* **Nano comparison:** FeatherLM Nano (13,239,168 params) vs. a Transformer with the same
  width/depth and FFN widened to 1,118 to match parameters (13,237,632). Identical data,
  tokenizer, sequence length 256, effective batch 16 sequences, AdamW (lr 1e-3, warmup 100,
  cosine), dropout 0.1, evaluation every 100 steps, early stopping with patience 3.
* **Multi-seed comparison:** four 4M-parameter architectures (Transformer with exactly
  matched parameters; Feather hybrid; pure recurrence; hybrid with sharing), 300 steps of
  16×128 tokens (614k tokens), 3 seeds each.
* **Metrics:** validation loss / perplexity, bits per byte, next-token top-1 accuracy, peak
  RSS, tokens/s, wall time, decode speed and cache size vs. context, quantization fidelity.
  Energy is not measured (no power counters on the VM).

## 6. Results

All numbers come from `results/*.json` (full tables in `docs/benchmarks.md`).

### 6.1 Training works on a CPU

FeatherLM Nano trained from random weights on 2 vCPUs: training loss fell from 9.01 to 3.08
in 600 steps (2.46M tokens, 40 min, mean 1,415 tok/s, measured peak RSS 2,154 MB). Validation
loss bottomed at step 300 and early stopping ended the run at step 600: the model overfits
the 300k-token corpus.

### 6.2 Nano vs. matched Transformer (context 256, one seed)

| | Feather Nano | Transformer |
|---|---|---|
| Parameters | 13,239,168 | 13,237,632 |
| Best validation loss (evaluate.py) | 5.131 | **5.022** |
| Perplexity / bits per byte | 169.2 / 2.360 | **151.7 / 2.310** |
| Next-token top-1 accuracy | 22.7% | **23.9%** |
| Training throughput | 1,415 tok/s | **1,536 tok/s** |
| Peak training RSS | 2,154 MB | **2,077 MB** |

**At this scale and context length the Transformer is better on every training measure.**
With a single seed the loss gap (0.11 nats) is suggestive rather than conclusive, but it does
not support a claim that the Feather core is more sample-efficient.

### 6.3 Multi-seed comparison (4.1M parameters, context 128, 3 seeds)

| Architecture | Params | Val loss (mean ± std) | Train tok/s | Peak RSS |
|---|---|---|---|---|
| Transformer (matched) | 4,100,544 | 5.389 ± 0.046 | 3,264 | 1,414 MB |
| Feather hybrid | 4,100,544 | 5.360 ± 0.036 | 3,636 | 1,472 MB |
| Feather recurrent only | 4,183,488 | 5.345 ± 0.019 | 3,705 | 1,477 MB |
| Feather hybrid, shared weights | 3,230,400 | 5.376 ± 0.041 | 3,594 | 1,464 MB |

The Feather variants average slightly lower loss, but the differences (0.01-0.04 nats) are
within one standard deviation: **no quality difference can be claimed**. Consistent across
seeds: Feather trained 10-14% faster at this small width and used ~60 MB more peak RAM.
Weight sharing kept quality level with the Transformer using 21% fewer parameters, which is
the most interesting signal for memory-constrained deployment and worth a larger study.
That Nano reversed the ranking (6.2) shows these results do not transfer across scales
without further evidence.

### 6.4 Inference memory and speed (Nano configs, random weights)

| Context | Feather cache | Transformer KV cache | Feather decode | Transformer decode |
|---|---|---|---|---|
| 256 / 128 | 0.146 MB | 1.5 MB | 213.5 tok/s | 188.4 tok/s |
| 1,024 / 512 | 0.146 MB | 6.0 MB | 202.6 tok/s | 171.5 tok/s |
| 4,096 / 2,048 | 0.146 MB | 24.0 MB | 188.8 tok/s | 117.7 tok/s |

(Cache columns use contexts 256/1,024/4,096; speed columns use 128/512/2,048.)

This is the clearest positive result: **inference state is constant** (0.146 MB) while the
Transformer's grows linearly, and decode speed degrades far less with context (60% faster
at 2,048 tokens). The trade-off is prefill: time to first token at 2,048 tokens is 397 ms vs.
312 ms, because the recurrence runs as a plain PyTorch scan while attention uses fused kernels.

Training memory is the opposite of the project's hope: without checkpointing Feather Nano
peaked at 1,913 MB vs. 1,701 MB (the scan stores extra intermediates). Activation
checkpointing brings both to ~1,250 MB, within the 4 GB tier.

### 6.5 Quantization (Nano)

| Format | File | Perplexity | Top-1 agreement with FP32 |
|---|---|---|---|
| FP32 | 50.5 MB | 174.04 | 100% |
| FP16 | 25.3 MB | 174.04 | 99.95% |
| INT8 (per-channel) | 12.8 MB | 173.98 | 99.6% |
| INT4 (group 32) | 7.2 MB | 177.41 | 89.6% |

INT8 is effectively lossless here; INT4 costs 2% perplexity. Keeping weights packed in memory
cuts resident weights by 57-65% but slows decoding to about half (INT8) or a third (INT4) without native low-bit kernels.
ONNX export matches PyTorch to 2.3e-5.

### 6.6 Summary of evidence

* Supported: CPU-only training and inference of a small model with measured memory; constant
  inference state and context-robust decode speed for the Feather backend; near-lossless INT8.
* Not supported: better quality per parameter or lower training memory than a matched
  Transformer. The Nano result points the other way.
* Open: whether weight sharing and constant state give a better quality-per-memory trade-off
  at larger scale and longer contexts.

## 7. Limitations

* One tiny corpus; the 13M-parameter models overfit within a few hundred steps, so the Nano
  comparison measures sample efficiency on small data more than capacity.
* Small budgets and few seeds; no downstream task evaluation beyond next-token metrics.
* Wall-clock comparisons depend on implementation maturity: the attention path uses fused
  PyTorch kernels while the recurrence is a plain-PyTorch scan.
* Long-context behaviour, where constant inference state matters most, was measured for
  memory and speed with random weights but not for quality.
* No energy measurements.

## 8. Future work and experiments for a paper

1. **Scale and data:** 30M-350M parameters on 1B+ tokens of a licensed corpus (e.g. a
   permissively licensed web or book corpus), with compute-matched (FLOPs) and
   memory-matched comparisons, not only parameter-matched.
2. **Ablations:** low-rank vs. full gates; conv width; attention frequency (every 2/3/4/∞
   layers); window size; sharing ratio; factorized embeddings.
3. **Long context:** perplexity vs. position up to 8-32k tokens, passkey retrieval and
   associative recall, where recurrent models are known to struggle.
4. **Systems:** fused CPU scan kernel; INT8 matmul runtime; stateful ONNX export; measured
   energy per token on laptops (RAPL) and phones (battery APIs).
5. **Baselines:** compare with published efficient architectures (Mamba, RWKV, Griffin
   reproductions) at matched budgets.
6. **Statistics:** at least 3-5 seeds per configuration with confidence intervals.

A paper would need the scale-up and ablations above; the current evidence is not publication-ready.
