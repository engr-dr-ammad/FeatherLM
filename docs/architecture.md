# FeatherLM architecture

*Developed by Ammad Aslam with AI-assisted tooling.*

FeatherLM is a decoder-only autoregressive language model with two interchangeable
sequence-mixing backends that share the tokenizer, embeddings, feed-forward layers,
training loop and inference stack.

```
token ids ─▶ TokenEmbedding (tied, optionally factorized) ─▶ L × Block ─▶ RMSNorm ─▶ tied vocabulary projection ─▶ logits
                      Block = x + Mixer(RMSNorm(x));  x + SwiGLU(RMSNorm(x))
```

| Backend | Mixer per layer | Inference state | Code |
|---|---|---|---|
| `transformer` | global causal GQA self-attention + RoPE | KV cache, grows with context | `featherlm/model/attention.py` |
| `feather` | gated linear recurrence; every `attn_every`-th layer uses sliding-window attention | fixed size: recurrent state + windowed KV cache | `featherlm/model/recurrent.py` |

## What is established and what is new

Almost every component is an established technique. FeatherLM does not claim them as inventions.

| Component | Status | Reference |
|---|---|---|
| Byte-level BPE tokenizer, GPT-2 style pre-tokenizer | established | Radford et al. 2019 |
| RMSNorm | established | Zhang & Sennrich 2019 |
| SwiGLU feed-forward | established | Shazeer 2020 |
| Rotary position embeddings | established | Su et al. 2021 |
| Grouped-query attention | established | Ainslie et al. 2023 |
| Weight tying of input/output embeddings | established | Press & Wolf 2017 |
| Factorized embeddings | established | Lan et al. 2019 (ALBERT) |
| Cross-layer parameter sharing | established | Dehghani et al. 2018; Lan et al. 2019 |
| RG-LRU gated linear recurrence | established | De et al. 2024 (Griffin/Hawk) |
| Short causal depthwise conv before recurrence | established | H3, Mamba, Griffin |
| Hybrid recurrence + local attention | established | Griffin; Mistral-style sliding windows |
| LoRA | established | Hu et al. 2021 |
| **Low-rank gate projections in the recurrence** (d → d/4 → 2d) | implementation choice / hypothesis | this project |
| **Chunked log-space scan with bounded decay** (O(T·d) activation memory, no T×T or C×C×d tensors) | implementation choice | this project |
| **Combination tuned for CPU-only training under an explicit RAM budget**, with sharing as a knob | research hypothesis, not yet supported by evidence | this project |

## Mathematical formulation

### Gated recurrence block (Feather backend)

For input $x_t \in \mathbb{R}^d$ (after RMSNorm):

$$[u_t, g_t] = W_{in} x_t \qquad u_t \leftarrow \text{CausalDepthwiseConv}_K(u)_t$$

$$[r_t, i_t] = \sigma\!\left(W_{up}\, W_{down}\, u_t + b\right) \qquad W_{down} \in \mathbb{R}^{d/4 \times d}$$

$$\log a_t = \max\!\left(-c \cdot \text{softplus}(\Lambda) \odot r_t,\ -4\right), \quad c = 8$$

$$h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot u_t)$$

$$y_t = W_{out}\left(h_t \odot \text{GELU}(g_t)\right)$$

$\Lambda$ is initialised so $a_t \in [0.9, 0.999]$ when $r_t = 1$. Every operation is
either position-wise or depends only on earlier positions, so the block is causal by
construction; `tests/test_model.py::test_causal_masking` verifies this numerically.

### Chunked scan

The linear recurrence $h_t = a_t h_{t-1} + b_t$ is evaluated in chunks of $C \le 20$ steps.
Inside a chunk that starts with carried state $h_0$, with $A_t = \sum_{r \le t} \log a_r$:

$$h_t = e^{A_t}\left(h_0 + \sum_{s \le t} e^{-A_s} b_s\right)$$

Both sums are `cumsum`s, so a chunk costs O(C·d) and stores O(C·d) activations. Because
$\log a_t \ge -4$ and $C \le 20$, the largest exponent is 80, below float32's limit of
about 88.7, so nothing overflows. The relative precision of $h_t$ is that of float32,
because the dominant term in the bracket is always the most recent one. A sequential
reference (`reference_scan`) is tested against it to 1e-4.

### Sliding-window attention

Feather attention layers use the same GQA + RoPE module as the baseline with the mask
$0 \le t - s < w$. During generation the KV cache keeps only the last $w-1$ entries.

## Complexity

Notation: $L$ layers, $T$ sequence length, $d$ model width, $d_{kv} = n_{kv}\cdot d/n_h$,
$f$ = FFN width, $w$ = local window, $V$ vocabulary, $e$ embedding width.

### Parameters per block

| Part | Parameters |
|---|---|
| GQA attention | $2d^2 + 2 d\, d_{kv}$ |
| Gated recurrence | $3d^2 + 3d\cdot\tfrac{d}{4} + d(K + 4) \approx 3.75 d^2$ |
| SwiGLU FFN | $3 d f$ |
| Embeddings (tied) | $V d$, or $V e + e d$ when factorized |

With $n_h = 6$, $n_{kv} = 2$ the attention mixer has $2.67 d^2$ parameters, so a recurrent
layer is about 40% larger than an attention layer. The benchmark configs compensate by
widening the Transformer's FFN so the totals match (see `configs/bench_micro.yaml`).
Depth-wise sharing with $u$ unique blocks divides block parameters by roughly $L/u$
without reducing compute.

### Compute and memory

| | Transformer | Feather (hybrid) |
|---|---|---|
| Training FLOPs / token | $O(L(d^2 + d f) + L\,T\,d)$ | $O(L(d^2 + d f) + L_{attn}\, w\, d)$ |
| Training activations | $O(L T d)$ + attention scratch | $O(L T d)$ (scan adds a constant factor) |
| Prefill time | $O(L T^2 d)$ attention term | $O(L T d^2)$, attention term $O(L_{attn} T w d)$ |
| Decode time / token | $O(L(d^2 + T d))$, grows with context | $O(L d^2)$, constant |
| Inference state | $2 L T d_{kv}$ floats, grows with context | $L_{rec}(d + (K-1)d) + 2 L_{attn}(w-1) d_{kv}$ floats, constant |
| Optimizer state (AdamW, fp32) | $2\times$ parameters | same |

The constant decode state is the main property of interest for phones and small CPUs.
At Nano's sizes, $T = 256$ is short enough that the Transformer's cache is still small; the
difference grows with context (measured in `results/memory.json`).

## Inference paths

* `model(idx, caches=model.new_cache())` returns per-layer state for incremental decoding.
  Tests check incremental decoding matches a full forward pass for both backends.
* The Transformer crops and re-encodes its context when it reaches `max_seq_len`; the Feather
  backend continues with constant memory (quality beyond the training length is not evaluated).
* ONNX export uses a fixed window with right padding; see `featherlm/export/onnx_export.py`.

## Memory-efficiency features

| Feature | Status |
|---|---|
| Gradient accumulation | implemented (`training.grad_accum`) |
| Activation checkpointing | implemented (`model.gradient_checkpointing`), gradients tested equal |
| Parameter sharing across depth | implemented (`model.n_unique_blocks`) |
| Reduced optimizer state | Adafactor option (`training.optimizer: adafactor`) |
| bf16 autocast | implemented (`training.precision: bf16`); speed depends on CPU bf16 support |
| Low-bit inference | FP16 / INT8 / INT4 weight-only (`featherlm/export/quantize.py`) |
| Memory-efficient attention | PyTorch `scaled_dot_product_attention` (fused kernels where available) |
| Short-context presets | `configs/nano_cpu.yaml` (T = 256) |
| Streaming datasets | `np.memmap` token streams: dataset size does not affect RAM |
| Memory budget | pre-flight estimate vs. *available* RAM minus a reserve, plus peak-RSS monitoring |
