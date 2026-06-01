# 2026-06-01 Qwen Unit Math Oracle

## Code And Data Changed

- Added `qwen_unit_math_oracle` to the Qwen persistent task-body manifest.
- The oracle records a hidden-size-4 reference for RMSNorm, projection,
  single-token attention cache writeback, SiLU/SwiGLU, and logits equations.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-task-bodies-qwen-unit-2026-06-01/`
  `qwen-persistent-task-bodies.json`.

## Architecture Quality

The task-body evidence now separates three claims: source generation through
the persistent DAG generator, deterministic proxy arithmetic, and real Qwen
unit equations. A later source-coverage change adds generated CUDA source for
the same unit equations; this report records the oracle-introduction step.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `1 passed`.

## Remaining Gaps

- Execute the Qwen unit-math source path through `cuda_live`.
- Execute the full Qwen decode loop and import full-serving viewer rows.
