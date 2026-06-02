# Qwen Attention Output Projection

## Code And Data Changed

- Updated `qwen_attention_o` so the main KV-cache attention path consumes
  `o_proj_weight` through `tensor_args[0]` and writes projected hidden columns
  instead of raw attention values when the projection weight is bound.
- Added `qwen_attention_o_bounded_projection_source` as explicit source
  evidence and declared `scalar_args` in the task-body manifest because
  `scalar_args[1]` bounds the diagnostic projection input width.
- Updated unit-math launch packets so `qwen_attention_o` uses a bounded
  projection width in resource-backed diagnostics.
- Linked compact source and live diagnostic raw artifacts from the paper
  evaluation matrix. The generated CUDA/PTX artifacts remain under `tmp/`.

## Architecture Quality

The attention-output task now maps closer to the real Qwen decode layer:
attention context is computed over the mutable paged KV cache, then multiplied
by the output projection weight. The projection remains bounded for diagnostic
execution so persistent-device smoke tests can exercise the path without
pretending to be full Qwen numerical correctness.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py \
  tests/ut/py/test_nvidia_review_artifacts.py::test_qwen_persistent_task_bodies_render_generated_source \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-attention-o-projection/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-attention-o-projection/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-attention-o-projection/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-attention-o-projection/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-attention-o-projection/runner-cache \
  --output-json tmp/cuda-backend/qwen-attention-o-projection/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed and recorded unit-math full-RMSNorm
execution with weighted-elementwise branches.

## Remaining Gaps

This is bounded diagnostic execution. The next implementation step is to keep
replacing proxy Qwen math with real task bodies until full-serving PTO rows can
pass numerical correctness against MPK and VDCores policy baselines.
