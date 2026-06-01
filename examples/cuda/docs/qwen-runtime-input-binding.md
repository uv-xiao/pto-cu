# CUDA Examples: Qwen Runtime Input Binding

## Qwen Runtime Input Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_runtime_input_binding.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-input-binding/qwen-runtime-input-binding.json
```

Expected output: command exits 0; output JSON records padded target-length
Qwen `input_ids`, matching `attention_mask`, decode `output_ids` capacity,
prompt alignment status, and scalar bindings for the MPK and VDCores serving
policies.

This is a host-side runtime input artifact, not a CUDA allocation. It turns
the tokenizer output into `runtime_token_buffer_plan`,
`attention_mask_buffer`, and `decode_output_buffer_plan`; CUDA token-buffer
allocation and decode-loop consumption remain runtime gaps.

