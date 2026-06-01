# CUDA Examples: Qwen CUDA Weight Binding

## Qwen CUDA Weight Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --output-json tmp/cuda-backend/pto-serving-weight-binding/qwen-cuda-weight-binding.json
```

Expected output: command exits 0; output JSON records stable CUDA binding
slots, safetensors file byte ranges, persistent-device readonly weight
argument roles, and bounded or full CUDA residency probe status.

This is a binding artifact, not a full model loader. With local CUDA runtime
libraries available it copies a bounded subset of small tensors to device
memory through the existing runtime C API, then frees them. Full weight
residency can be probed explicitly:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --cuda-probe-mode full \
  --device 4 \
  --verify-tensors 16 \
  --output-json tmp/cuda-backend/pto-serving-weight-residency/qwen-cuda-weight-residency.json
```

The full mode keeps all copied tensors resident until the whole model is
loaded, verifies selected small tensors by copying bytes back, then frees every
allocation. Persistent task-argument pointer binding remains a runtime gap.

