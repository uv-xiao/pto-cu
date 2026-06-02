# 2026-06-02 C++ Snapshot Malformed Coverage

## Code And Data Changed

- Added `cuda_persistent_smoke_impl/snapshot_malformed.py` to exercise
  malformed normal-graph lowering from live C++ orchestrator snapshots.
- Added `cuda_snapshot_malformed.py` so the coverage can write a reviewable
  JSON artifact under `tmp/`.
- Added a focused unit test for the expected malformed snapshot failures.
- Removed persistent scheduler generalization from `status.md` remaining gaps.

## Architecture Quality

The malformed coverage reuses the live `_debug_next_level_submits()` path and
the existing `cuda_pto_graph.py` validation points. It stays separate from the
large smoke runner and records failures at the builder/lowering boundary rather
than introducing device-runtime behavior for invalid host inputs.

## Evaluation Run

Focused malformed snapshot coverage passed:

```bash
.venv/bin/python -m pytest -q tests/ut/py/test_cuda_backend.py \
  -k 'cpp_snapshot_malformed_cases'
```

Review artifact:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_snapshot_malformed.py \
  --output-json \
  tmp/cuda-backend/cpp-orchestrator-snapshot-malformed-326cb61/snapshot-malformed.json
```

The artifact records expected failures for:

- multi-`TaskArgs` snapshot entries;
- tensor-name arity mismatch;
- duplicate snapshot slot keys.

## Remaining Gaps

Backend implementation closure remains `in_progress` because tuned tensor
workloads are still listed under `status.md` remaining gaps, and final
paper-grade results still depend on the paper-readiness work queue.
