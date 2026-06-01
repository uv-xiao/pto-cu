# 2026-06-02 Qwen Resource-Backed Execution

## Code And Data Changed

- Added `--run-resource-backed-smoke` to the Qwen decode-loop runner.
- Added focused resource-backed execution modules for building device graph
  state from the launch packet and running it through `run_prepared`.
- Updated the CUDA example manifest, README, and benchmark-viewer matrix with
  the new diagnostic raw artifact.

## Architecture Quality

The Qwen runner now has an executable path after launch-packet preflight. It
keeps token buffers, KV-cache, resident weights, activation workspace, and the
runtime graph state inside the same CUDA context, then prepares the generated
Qwen persistent-device task-function set and submits both serving-policy DAGs
through `run_prepared`. The status remains diagnostic because the task bodies
are not yet full Qwen kernels and the row is not imported as full-serving
paper evidence.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --single-context-live-session --run-resource-backed-smoke \
  --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-resource-backed-execution-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `resource_backed_execution.status` is `pass`. Both
`mpk_offline_decode` and `vdcores_offline_decode` execute 255-task DAGs with
`run_prepared_status=0`, `completed_count=255`, and `error_count=0`. The
single-context session closes after freeing 955 pointers, including the graph
runtime-state allocations.

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q
```

Result: `4 passed`.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Import full-serving PTO rows for the MPK and VDCores serving policies.
- Keep VDCores and ThunderKittens baseline rows progressing in parallel.
