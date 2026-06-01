# CUDA Examples: Qwen Persistent Weight Arguments

## Qwen Persistent Weight Arguments

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_args.py \
  --output-json tmp/cuda-backend/pto-serving-weight-args/qwen-persistent-weight-args.json
```

Expected output: command exits 0; output JSON records Qwen weight task
descriptors whose `tensor_args` fit the four-pointer persistent DAG ABI and
cover every validated weight tensor.

This is an ABI manifest, not runtime pointer materialization. It decomposes
Qwen layer work into persistent task descriptors such as attention QKV,
attention Q/K norm, MLP gate/up, and MLP down so each task stays within
`PtoCudaPersistentDagTask::tensor_args[4]`.

