# 2026-06-01 Mutable KV-Cache ABI

## Code And Data Changed

- Changed persistent DAG `c` and `d` fields from `const float *` to `float *`
  in `src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h`.
- Updated `simpler_setup/cuda_callable_compiler.py` so generated persistent
  DAG source uses the same mutable `c`/`d` fields.
- Updated Qwen task-body evidence so attention task bodies write through
  `task->c[kv_index]` and `task->d[kv_index]`.
- Refreshed serving scaffold, preflight, paper-readiness matrix, audit, and
  work-queue data.

## Architecture Quality

This keeps the host ABI and generated CUDA device ABI aligned. The Qwen
task-body artifact now records mutable KV-cache field access without claiming
numerical Qwen correctness or live decode execution.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_persistent_codegen.py \
  -q -k 'mutable_kv_cache_fields or fourth_tensor_descriptor'
```

Result: the focused TDD test first failed while `c` and `d` were const, then
passed after the ABI and generator update.

## Remaining Gaps

- Replace source-level task bodies with numerically correct Qwen kernels.
- Execute the `cuda_live` decode loop and import full-serving viewer rows.
