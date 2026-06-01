# CUDA Examples: Qwen Token Pointer Table

## Qwen Token Pointer Table

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_token_pointer_table.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-token-pointers/qwen-token-pointer-table.json
```

Expected output: command exits 0; output JSON records token pointer-table
lifecycle evidence.

The default mode is a deterministic dry-run lifecycle for review and CI-free
checks. It keeps Qwen `input_ids`, `attention_mask`, and `output_ids` pointers
live while persistent decode args are materialized. Add `--cuda-live` to
allocate real CUDA token buffers through the host runtime, then close the
pointer table after decode argument materialization.

