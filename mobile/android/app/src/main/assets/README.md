Copy the ONNX export here before building:

    python -m featherlm.export.onnx_export --model runs/featherlm-nano/final --out models/featherlm-nano-onnx
    cp models/featherlm-nano-onnx/{model.onnx,tokenizer.json,onnx_config.json} mobile/android/app/src/main/assets/
