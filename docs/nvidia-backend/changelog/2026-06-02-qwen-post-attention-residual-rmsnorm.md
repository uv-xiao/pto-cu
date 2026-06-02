# Qwen Post-Attention Residual RMSNorm

## Code And Data Changed

- Updated `qwen_rmsnorm_post_attention` generated CUDA source to consume
  `task->b` as the attention residual source.
- Changed the post-attention RMSNorm reduction to compute mean square over
  `attention_output + residual` before applying
  `post_attention_layernorm.weight`.
- Added role-aware launch-packet binding so post-attention RMSNorm receives
  the input to the matching layer's input RMSNorm through `b`.
- Added `qwen_post_attention_residual_rmsnorm_source` as compact source
  evidence in the task-body and example manifests.

## Architecture Quality

This removes one diagnostic shortcut from the Qwen block lifecycle. The
post-attention norm task now models the Qwen residual edge explicitly through
the existing `CudaPersistentDagTask::b` field instead of normalizing only the
attention projection output. The launch-packet helper derives the residual
source from descriptor order, so no extra tensor-argument slot is taken from
the weight and runtime-table ABI.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_launch_packet_binds_post_attention_residual_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py \
  tests/ut/py/test_nvidia_review_artifacts.py::test_qwen_persistent_task_bodies_render_generated_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/runner-cache \
  --output-json tmp/cuda-backend/qwen-post-attention-residual-rmsnorm/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed with `resource_backed_execution.status`
set to `pass`.

## Remaining Gaps

This still is not full Qwen serving correctness. The MLP residual edge after
`qwen_mlp_down` still needs an explicit preserved residual source, and the
paper claim still requires imported full-serving PTO rows with numerical
correctness, latency, and throughput metrics.
