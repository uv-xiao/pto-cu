# Qwen Resource Prefix Execution

## Code And Data Changed

- Added `--resource-backed-max-tasks` to the Qwen decode-loop runner.
- The resource-backed executor now submits a prefix of the materialized Qwen
  descriptor sequence when this bound is set, while leaving the full
  materialization and preflight surfaces unchanged.
- Prefix runs that do not include the final `qwen_logits` task now record
  `bounded_prefix_without_logits_task` instead of attempting a logits read.
- Resource-backed execution metadata records `repeat_policy.max_task_count` so
  bounded evidence cannot be mistaken for a full-graph run.

## Architecture Quality

This adds an execution knob for real live CUDA DAG prefixes rather than another
static checker. It preserves the same resident weights, activation workspace,
host task packet builder, persistent-device PTX, and CUDA runtime entry point
used by the full Qwen resource-backed path.

The bounded prefix is explicitly diagnostic. It is useful for locating the
current scaling boundary in the persistent-device runner, but it does not
promote partial execution to full-serving evidence.

## Evaluation Run

Focused forwarding test passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q -k \
  resource_backed_smoke_runs_before_single_context_close
```

Live A100 prefix runs were written under
`tmp/cuda-backend/qwen-resource-timing/`:

| Max tasks | Status | Completed | Device time |
| --------- | ------ | --------- | ----------- |
| 1 | pass | 1 | 0.628 ms |
| 2 | pass | 2 | 0.874 ms |
| 4 | pass | 4 | 1347.241 ms |
| 8 | pass | 8 | 8559.606 ms |
| 16 | pass | 16 | 17205.723 ms |

The first 16 tasks run through `layer_2_input_norm`, proving that the live
persistent scheduler can execute real resource-backed Qwen DAG prefixes and
that the current full-graph timeout is dominated by scalar diagnostic math
scaling, not by immediate scheduler failure.

## Remaining Gaps

The full 255-task `mpk_offline_decode` graph still times out under the current
scalar diagnostic formulas. The next implementation target is replacing the
heavy scalar projection stages with tiled or tensor-core CUDA math so the same
resource-backed path can complete full Qwen diagnostic and then paper-grade
serving runs.
