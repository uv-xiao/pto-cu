# Qwen Post-Attention RMSNorm

## Code And Data Changed

- Changed `qwen_rmsnorm_post_attention` from an element-threaded weight
  multiply to a block-threaded RMSNorm body with two explicit branches:
  `scalar_arg_count == 1` runs the full hidden-vector reduction, while
  `scalar_arg_count > 1` keeps the external-scale weighted path.
- Updated `unit_math` and `unit_math_full_rmsnorm` launch-packet contracts so
  post-attention RMSNorm selects the intended branch.
- Added `qwen_post_attention_norm_full_rmsnorm_source` as a task-body evidence
  symbol and linked the new raw resource-backed diagnostic artifact from the
  paper evaluation matrix.

## Architecture Quality

This removes a Qwen layer-norm asymmetry in the persistent-device runtime:
input RMSNorm, post-attention RMSNorm, and final RMSNorm now share the same
block-threaded full-reduction contract instead of mixing reduction and
elementwise-only behavior.

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
  --output-json tmp/cuda-backend/qwen-post-attention-rmsnorm/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-post-attention-rmsnorm/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-post-attention-rmsnorm/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-post-attention-rmsnorm/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-post-attention-rmsnorm/runner-cache \
  --output-json tmp/cuda-backend/qwen-post-attention-rmsnorm/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the resource-backed
MPK and VDCores bounded diagnostics each completed 6 tasks with zero scheduler
errors and listed `qwen_rmsnorm_post_attention` in the full-reduction
contracts.

## Remaining Gaps

This is still diagnostic bounded execution. Paper readiness still requires full
Qwen numerical correctness and full-serving PTO rows for MPK and VDCores.
