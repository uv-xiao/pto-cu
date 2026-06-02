# Qwen QK Norm Block Threading

## Code And Data Changed

- Changed `qwen_attention_qk_norm` from element-threaded per-output reduction
  to a block-threaded row RMSNorm plus RoPE body.
- Added `qwen_qk_norm_block_rmsnorm_rope_source` as explicit task-body
  evidence.
- Linked the bounded resource-backed diagnostic artifact in the compact paper
  evaluation matrix.

## Architecture Quality

The QK norm/RoPE task now reduces the head vector once per device task and
writes outputs through a block-stride loop. This matches the persistent-device
scheduler model better than recomputing the same row reduction for every
element lane.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-qk-norm-block/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-qk-norm-block/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-qk-norm-block/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-qk-norm-block/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-qk-norm-block/runner-cache \
  --output-json tmp/cuda-backend/qwen-qk-norm-block/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded MPK and
VDCores diagnostics each completed 6 tasks with zero scheduler errors and
included `qwen_attention_qk_norm` in the numeric-ready path.

## Remaining Gaps

This remains bounded diagnostic execution. Full Qwen numerical correctness and
paper-ready PTO full-serving rows are still required.
