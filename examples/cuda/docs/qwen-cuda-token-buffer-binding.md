# CUDA Examples: Qwen CUDA Token Buffer Binding

## Qwen CUDA Token Buffer Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_token_buffer_binding.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-token-buffer/qwen-cuda-token-buffer-binding.json
```

Expected output: command exits 0; output JSON records CUDA allocation and copy
verification for Qwen input_ids, attention_mask, and output_ids token buffers
when the host runtime is available.

The artifact allocates those buffers, copies host token data to device memory,
verifies copy-back, and leaves decode-loop consumption as the remaining
runtime gap.

