# Qwen Worker Block Plumbing

## Code And Data Changed

- Added `--resource-backed-worker-blocks` to the Qwen decode-loop runner.
- Threaded the worker-block count through the single-context lifecycle into the
  live CUDA resource-backed executor.
- The executor now derives `grid_dim = scheduler_blocks + worker_blocks` before
  preparing the persistent-device callable. The default remains one scheduler
  block plus one worker block.
- `MaterializedGraph` now stores configurable scheduler-block and block-dim
  values, allocates one scheduler progress counter per scheduler block, and
  reports all scheduler progress slots.
- Resource-backed execution evidence now records `repeat_policy.scheduler_blocks`,
  `repeat_policy.worker_blocks`, and `repeat_policy.grid_dim`.
- No benchmark-viewer data was imported for this slice. The raw live run is kept
  under `tmp/cuda-backend/qwen-worker-blocks/`.

## Architecture Quality

This is real persistent-device runtime plumbing. It exposes the generated CUDA
executor's existing scheduler/worker partition through the Qwen runner instead
of adding a checker-only surface.

The current Qwen diagnostic DAG is mostly a dependency chain, so this does not
claim throughput improvement. The value is architectural: future tiled
projection, attention, and ring-buffer tasks can run with multiple resident
worker blocks without changing the host task packet ABI or writing separate
`__global__` kernels for each task body.

## Evaluation Run

Focused unit coverage passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q -k \
  'resource_backed_smoke_runs_before_single_context_close or materialized_graph_uses_configured_scheduler_blocks'
```

Live bounded A100 resource-backed smoke passed:

```bash
PYTHONPATH=$PWD:$PWD/python timeout 90 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-repeat-runs 1 \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-max-tasks 16 \
  --resource-backed-worker-blocks 4 \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-worker-blocks \
  --output-json tmp/cuda-backend/qwen-worker-blocks/qwen-mpk-prefix-16-workers4.json
```

Result summary:

| Field | Value |
| ----- | ----- |
| Status | pass |
| Workload | `mpk_offline_decode` |
| Prefix tasks | 16 |
| Scheduler blocks | 1 |
| Worker blocks | 4 |
| Prepared grid dim | 5 |
| Completed count | 16 |
| Error count | 0 |

## Remaining Gaps

Multiple worker blocks are now configurable and launchable, but the current
diagnostic prefix remains serially constrained by DAG dependencies and scalar
formula bodies. The next useful runtime work is tiled projection or attention
tasks that create independent ready work for those resident worker blocks.
