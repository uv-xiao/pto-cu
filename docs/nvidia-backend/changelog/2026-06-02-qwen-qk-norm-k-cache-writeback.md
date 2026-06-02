# 2026-06-02 Qwen QK Norm K-Cache Writeback

## Code And Data Changed

- Added descriptor page-size metadata for `qwen_attention_qk_norm` so the
  generated CUDA body can map the normalized K region back into KV-cache
  layout.
- Updated the QK norm generated source to consume mutable `task->c` key-cache
  storage and write normalized K-region outputs back using decode-position
  scalar metadata.
- Added `qwen_qk_norm_normalized_k_cache_writeback_source` as explicit source
  evidence in the task-body manifest, example manifest, and review matrix.

## Architecture Quality

- Keeps Q and K computation in one QK-norm callable while avoiding a separate
  host-launched CUDA kernel for key-cache mutation.
- Preserves the current four-slot tensor argument contract: Q norm, K norm,
  RoPE cosine, and RoPE sine. Because no tensor slot remains for a page table,
  this diagnostic path uses identity logical-to-physical page mapping.
- Makes the key-cache side effect visible through `consumes_fields`, so review
  evidence can distinguish pure workspace output from cache mutation.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_qwen_weight_descriptors_emit_callable_shape_fields \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py \
  tests/ut/py/test_nvidia_review_artifacts.py::test_qwen_persistent_task_bodies_render_generated_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/runner-cache \
  --output-json tmp/cuda-backend/qwen-qk-norm-k-cache-writeback/qwen-decode-loop-runner.json
```

- Ran focused descriptor and generated-source tests for the new shape field and
  source evidence.
- Ran the adjacent Qwen CUDA example suite covering task-body math,
  graph-materialization descriptors, viewer import, and source rendering.
- Generated CUDA task bodies into `tmp/` and compiled them with `nvcc` for
  `compute_80`.
- Ran a bounded one-step resource-backed live diagnostic with resident weights,
  token, KV-cache, and workspace CUDA resources enabled.

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed with `resource_backed_execution.status`
set to `pass`.

## Remaining Gaps

This is not full serving correctness. The QK norm cache writeback currently
uses identity page mapping because the task's tensor argument slots are already
occupied by norm and RoPE tables. Full paged-cache correctness still needs a
larger task-argument contract or a scheduler-side way to provide page-table
metadata without displacing required QK norm inputs.
