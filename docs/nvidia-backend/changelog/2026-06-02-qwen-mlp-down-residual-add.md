# Qwen MLP Down Residual Add

## Code And Data Changed

- Updated `qwen_mlp_down` generated CUDA source to consume `task->b` as the
  launch-packet residual source and add it after the down projection.
- Added launch-packet binding for `qwen_mlp_down` so `b` points to the
  matching layer-local residual source instead of the token-side placeholder.
- Added `qwen_mlp_down_residual_add_source` as compact source evidence in the
  task-body and example manifests.
- Added resource-backed task coverage metadata so the bounded diagnostic
  records both callable order and Qwen func-id order.
- Linked generated-source and bounded live-diagnostic raw artifacts from the
  paper evaluation matrix while keeping raw files under `tmp/`.

## Architecture Quality

The MLP down task now models the second residual edge in a Qwen decoder block
through the existing `CudaPersistentDagTask::b` field. A later
`2026-06-03-qwen-mlp-residual-stream.md` changelog refines this contract so
`qwen_mlp_down` reconstructs the full pre-MLP residual stream from `task->b`
and runtime `tensor_args[1]`. The residual pointer is derived from the
layer-local descriptor order, matching the existing activation workspace
layout.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_launch_packet_binds_mlp_down_residual_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py \
  tests/ut/py/test_nvidia_review_artifacts.py::test_qwen_persistent_task_bodies_render_generated_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-mlp-down-residual-add/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-mlp-down-residual-add/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-mlp-down-residual-add/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-mlp-down-residual-add/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 8 --resource-backed-worker-blocks 8 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-mlp-down-residual-add/runner-cache \
  --output-json tmp/cuda-backend/qwen-mlp-down-residual-add/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed with `resource_backed_execution.status`
set to `pass`. The diagnostic recorded `task_coverage.task_count=8`,
`func_id_sequence=[7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107]`, and
included `qwen_mlp_down` in the callable sequence.

## Remaining Gaps

This still is not full Qwen serving correctness. The residual pointer is
launch-bound through the current activation graph, but paper readiness still
requires full-serving PTO rows with end-to-end numerical correctness,
latency, and throughput metrics for the selected MPK and VDCores policies.
