# 2026-06-02 Qwen Activation Workspace

## Code And Data Changed

- Added the Qwen activation/logits workspace lifecycle under the decode-loop
  runner.
- Added `--workspace-cuda-live` to allocate per-workload float32 activation
  buffers and a logits/sampling output buffer during launch-packet preflight.
- Updated the benchmark matrix, README, example manifest, and paper-readiness
  audit with the new raw artifact.

## Architecture Quality

The launch-packet preflight now separates resource binding from execution. With
the workspace owner enabled, task 0 consumes token ids, intermediate tasks read
the previous activation buffer and write the next activation buffer, and the
final task writes a float logits/sampling output buffer. The artifact still
records `execution_status=not_launched`, so the docs do not promote this to
full Qwen serving execution.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-submission-smoke --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --workspace-cuda-live --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-activation-workspace-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `activation_workspace_lifecycle.status=`
`activation_workspace_lifecycle_ready`, with 510 CUDA workspace pointers
allocated and freed. Both serving workloads report
`resource_backed_launch_packet_workspace_bound`, empty
`missing_runtime_buffers`, and 254 activation buffers bound into the packet.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Keep workspace, token, KV, and resident-weight owners open through actual
  `run_prepared` execution.
- Import full-serving PTO rows for the MPK and VDCores serving policies.
