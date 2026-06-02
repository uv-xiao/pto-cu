# Qwen QK Norm Separate Regions

## Code And Data Changed

- Updated `qwen_attention_qk_norm` descriptor shape fields to expose query
  heads, Q-width plus KV-width, head dimension, and KV-head count.
- Added a shape-aware generated CUDA branch that normalizes Q and K regions
  with their separate norm-weight slots before applying pairwise RoPE.
- Added `qwen_qk_norm_separate_qk_regions_source` as explicit source evidence.
- Linked compact source and live diagnostic raw artifacts from the paper
  evaluation matrix. The generated CUDA/PTX artifacts remain under `tmp/`.

## Architecture Quality

The QK norm task no longer represents the shape-aware path as one vector with
averaged Q/K norm weights. The generated source now distinguishes Q and K
regions in the packed QKV output, which is a closer match to Qwen decode math
and prepares the later key-cache normalized writeback work.

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
  --output-json tmp/cuda-backend/qwen-qk-norm-separate-regions/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-qk-norm-separate-regions/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-qk-norm-separate-regions/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-qk-norm-separate-regions/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-qk-norm-separate-regions/runner-cache \
  --output-json tmp/cuda-backend/qwen-qk-norm-separate-regions/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed and recorded unit-math full-RMSNorm
execution with weighted-elementwise branches.

## Remaining Gaps

This does not yet prove normalized K values are written back into the paged
KV cache for full serving. Full Qwen correctness still requires end-to-end
decode rows with real model outputs and latency/throughput metrics for PTO,
MPK, VDCores, and the paper baselines.
