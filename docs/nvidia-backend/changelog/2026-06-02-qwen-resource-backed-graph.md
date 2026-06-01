# 2026-06-02 Qwen Resource-Backed Graph

## Code And Data Changed

- Added a compact `resource_backed_graph_materialization` contract to the
  Qwen decode-loop runner.
- The contract binds live token fields, KV-cache fields, and resident
  weight pointers into the full 255-task Qwen DAG descriptor set for both
  serving policies.
- Added the new raw artifact to the LLM-serving paper evaluation matrix and
  refreshed the CUDA example manifest and README.

## Architecture Quality

The runner now preserves the resource-owner evidence needed immediately before
full `run_prepared` decode-loop execution: token `a/b/out`, KV `c/d`, and all
resident-weight tensor slots are checked together against the same
`CudaPersistentDagTask` ABI. The artifact emits task summaries instead of all
task structs, so reviewers can inspect binding quality without a large JSON
dump.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-submission-smoke --token-cuda-live --kv-cuda-live \
  --resident-cuda-live --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-resource-backed-graph-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `resource_backed_graph_materialization.status=`
`resource_backed_graph_materialized`, with two workload records, 255 graph
tasks per workload, 399 resident weight pointers, three token fields, two
KV fields, and zero missing pointers.

## Remaining Gaps

- Execute the resource-backed graph through `run_prepared`.
- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Import full-serving PTO rows for the MPK and VDCores serving policies.
