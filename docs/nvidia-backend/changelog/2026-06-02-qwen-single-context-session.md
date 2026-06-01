# 2026-06-02 Qwen Single Context Session

## Code And Data Changed

- Added `--single-context-live-session` to the Qwen decode-loop runner.
- Split the session implementation into focused runner, allocator, and
  submission-plan modules under `examples/cuda/qwen_decode_loop_runner_impl/`.
- Updated the CUDA example manifest, README, and benchmark-viewer matrix with
  the single-context raw artifact.

## Architecture Quality

The Qwen runner now has an explicit single CUDA-context owner session for the
persistent-device preflight path. Token buffers, KV-cache buffers, resident
weights, and activation workspace are allocated in that context, the
resource-backed graph and launch packets are materialized while those pointers
are live, and the session then frees each owner group in reverse lifetime
order. This keeps the document claim narrower than full execution: it proves
resource colocation and launch-packet binding, not numerical Qwen serving.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-submission-smoke --single-context-live-session \
  --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-single-context-session-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `single_context_live_session.status` is
`single_context_launch_packet_session_ready`, both serving workloads report
`resource_backed_launch_packet_workspace_bound`, and the session frees 935
CUDA pointers across token, KV-cache, resident-weight, and
activation-workspace owner groups.

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q
```

Result: `9 passed`.

## Remaining Gaps

- Execute the full resource-backed Qwen decode loop through `run_prepared`.
- Replace the diagnostic Qwen task bodies with numerically correct kernels.
- Import full-serving PTO rows for the MPK and VDCores serving policies.
