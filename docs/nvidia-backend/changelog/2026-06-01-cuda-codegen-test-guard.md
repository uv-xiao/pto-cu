# 2026-06-01 CUDA Codegen Test Guard

## Code And Data Changed

- Updated `tests/ut/py/test_cuda_persistent_codegen.py` so the scheduler
  error-source test checks the current ready-queue based persistent DAG
  initialization contract.
- Removed stale assertions for the deleted local `initial_ready_count`
  implementation detail.

## Architecture Quality

The persistent-device codegen guard now matches the scheduler shape that is
actually generated: scheduler blocks publish zero-fan-in tasks into the shared
ready queue, rendezvous through `scheduler_init_count`, and report empty or
unreachable graphs through queue and completion counters. This keeps the test
as contract evidence instead of preserving a removed implementation detail.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_persistent_codegen.py \
  -q
```

Result: `22 passed`.

## Remaining Gaps

- Full-serving Qwen evidence still needs numerically correct task bodies,
  `cuda_live` decode-loop execution, and viewer result import.
