# FeatherLM on mobile

*Developed by Ammad Aslam with AI-assisted tooling.*

Inference first; on-device training is a separate, experimental milestone.

## Status

| Item | Status |
|---|---|
| ONNX export (fixed window) | **tested**: matches PyTorch logits to < 1e-3 in ONNX Runtime 1.29 (desktop CPU), right-padding verified |
| Mobile inference algorithm | **tested on desktop**: `reference_runtime.py` reproduces the app loop with numpy + onnxruntime; greedy output matches the PyTorch engine (`tests/test_mobile_reference.py`) |
| Android app (`android/`) | source complete, **not built**: no Android SDK in the development environment |
| iOS package (`ios/FeatherLMKit`) | source complete, **not built or run**: requires Xcode on macOS and an iPhone |
| On-device training | not implemented (see below) |

Nothing here claims a successful native deployment. Build and measure on real devices
before relying on these numbers.

## Export the model

```bash
python -m featherlm.export.onnx_export --model runs/featherlm-nano/final --out models/featherlm-nano-onnx
python mobile/reference_runtime.py --onnx models/featherlm-nano-onnx --prompt "ROMEO:" --tokens 40
```

The export is a fixed-window graph: `input_ids (1, W) → logits (1, W, V)`. Prompts are
right-padded; because the model is causal, padding never changes the logits at real
positions. Each token re-runs the window, which is simple and portable but O(W) per token.
**Planned optimisation:** a stateful single-step export with the recurrent state and the
windowed KV cache as graph inputs/outputs. The Feather backend's state is fixed-size, which
makes this export straightforward compared with a growing Transformer KV cache.

Size: Nano is 13.2M parameters, about 53 MB in FP32 ONNX. ONNX Runtime's own
INT8 dynamic quantization (`onnxruntime.quantization.quantize_dynamic`) is the expected next
step for mobile size; the repository's INT8/INT4 PyTorch checkpoints are separate.

## Android

```bash
cp models/featherlm-nano-onnx/{model.onnx,tokenizer.json,onnx_config.json} mobile/android/app/src/main/assets/
cd mobile/android && gradle wrapper && ./gradlew installDebug    # needs Android SDK 35, JDK 17
```

* Runtime: `com.microsoft.onnxruntime:onnxruntime-android` (ARM64 CPU kernels). NNAPI/XNNPACK
  execution providers can be enabled in `FeatherModel.kt` via `SessionOptions` once tested.
* `FeatherTokenizer.kt` is a port of the Python byte-level BPE (same regex, NFC normalisation).
* The app has no INTERNET permission: generation is fully local.

## iOS

Add `mobile/ios/FeatherLMKit` as a local Swift package in Xcode, bundle `model.onnx`,
`tokenizer.json` and `onnx_config.json`, then:

```swift
let tok = try FeatherTokenizer(json: Data(contentsOf: Bundle.main.url(forResource: "tokenizer", withExtension: "json")!))
let model = try FeatherModel(modelPath: Bundle.main.path(forResource: "model", ofType: "onnx")!,
                             configURL: Bundle.main.url(forResource: "onnx_config", withExtension: "json")!, tokenizer: tok)
try model.generate(prompt: "ROMEO:") { id in print(tok.decode([id]), terminator: "") }
```

* Runtime: ONNX Runtime via Swift Package Manager. The Core ML execution provider (Neural
  Engine / GPU via Metal) can be appended to the session options; operator coverage for the
  recurrence ops must be checked on device.
* Memory: iOS terminates apps that exceed their memory limit (often 2-3 GB on 4-6 GB phones,
  less in extensions). Nano in FP32 needs roughly 60 MB of weights plus runtime buffers.

## On-device training (research notes)

Not implemented. The practical first step is LoRA adapter fine-tuning of Nano, which keeps
trainable state under 1 MB (see `featherlm/training/lora.py`). Issues to measure before
building it: ONNX Runtime Training or ExecuTorch operator support for the backward pass,
thermal throttling over minutes of sustained load, battery drain, and background execution
limits (Android WorkManager constraints, iOS BGProcessingTask time limits). Full pretraining
of a useful language model on a phone is not a goal.
