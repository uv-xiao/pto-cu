# 2026-06-01 Qwen Unit Math Source Coverage

## Code And Data Changed

- Added opt-in Qwen unit-math source paths to the generated persistent task
  bodies for RMSNorm, QKV cache writeback, SiLU/SwiGLU, and logits.
- Added `qwen_unit_math_source_coverage` to the task-body manifest evidence.
- Refreshed the task-body raw artifact under
  `tmp/cuda-backend/pto-serving-task-bodies-qwen-unit-2026-06-01/`.

## Architecture Quality

The generated source now contains the same hidden-size-4 Qwen unit equations
recorded by the oracle. Existing proxy live diagnostics keep their old path
until launch descriptors opt into unit math through scalar metadata.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `2 passed`.

## Remaining Gaps

- Execute the Qwen unit-math source path through `cuda_live`.
- Execute the full Qwen decode loop and import full-serving viewer rows.
