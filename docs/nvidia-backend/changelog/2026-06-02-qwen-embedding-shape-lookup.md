# Qwen Embedding Shape Lookup

## Code And Data Changed

- Updated `qwen_embedding_lookup` so bound descriptors use `cols` and `ldb`
  to index `embed_tokens.weight` by `(token_id, hidden_col)`.
- Added `qwen_embedding_shape_lookup_source` as explicit generated-source
  evidence.
- Linked compact source and live diagnostic raw artifacts from the paper
  evaluation matrix. The generated CUDA/PTX artifacts remain under `tmp/`.

## Architecture Quality

The persistent-device Qwen DAG now starts from the runtime token id and writes
the corresponding hidden-vector embedding columns instead of using the old
four-value proxy index. This moves another decode-stage task from scaffold
math toward the real Qwen serving path while preserving the fallback branch
for unbound diagnostic descriptors.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-embedding-shape-lookup/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-embedding-shape-lookup/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-embedding-shape-lookup/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-embedding-shape-lookup/qwen-persistent-task-bodies.ptx
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-max-tasks 6 --resource-backed-worker-blocks 6 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-embedding-shape-lookup/runner-cache \
  --output-json tmp/cuda-backend/qwen-embedding-shape-lookup/qwen-decode-loop-runner.json
```

Result: tests passed; generated source compiled to PTX; the bounded
resource-backed diagnostic completed and recorded unit-math full-RMSNorm
execution with weighted-elementwise branches.

## Remaining Gaps

This is bounded diagnostic execution. Full Qwen correctness still requires
end-to-end decode rows with real model outputs and latency/throughput metrics
for PTO, MPK, VDCores, and the paper baselines.
