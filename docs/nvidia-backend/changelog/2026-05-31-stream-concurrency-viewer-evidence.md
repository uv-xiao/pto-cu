# 2026-05-31 Stream Concurrency Viewer Evidence

## Code And Data Changed

- Added `host_schedule_stream_concurrency` viewer records for A100 and H200.
- Imported `pto_stream_serial` and `pto_stream_parallel` result records from
  the 10-repeat paired stream-pool capture.
- Updated the paper evaluation matrix so host-schedule launch-overhead evidence
  includes stream-concurrency rows and the new combined raw artifact.
- Regenerated `paper_readiness_audit.json` after reducing the host-schedule
  launch-overhead blocker to graph-replay sweeps.

## Architecture Quality

The imported records use the same JSON-backed viewer contract as the vector,
tensor, persistent, and paper-baseline rows. The raw artifact remains under
`tmp/`, while committed viewer data stores only stable repo-relative artifact
paths, statistics, method IDs, workload IDs, and evidence references.

The H200 run used the remote tree-sync path with `PTO_SOURCE_COMMIT=02bca4df`,
so the raw artifact records the source commit being evaluated even though no
remote Git fetch was required.

## Evaluation Run

Focused TDD checks for the stream-concurrency viewer mapping passed before the
capture commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_pair_stream_benchmark_builds_a100_h200_workflow \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_pair_stream_benchmark_merge_command_records_sanitized_examples \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_pair_stream_benchmark_sync_remote_tree_records_source_commit \
  tests/ut/py/test_nvidia_review_artifacts.py::test_cuda_viewer_export_generates_contract_records \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  -q
```

Result: `5 passed in 0.18s`.

The paired A100/H200 capture validated 40 raw rows:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_stream_benchmark.py \
  --branch goal/nvidia-paper-ready --repeats 10 --stream-pool-size 6 \
  --sync-remote-tree
```

Raw artifacts:

- `tmp/cuda-backend/a100-stream-pool6-02bca4df/`
- `tmp/cuda-backend/h200-stream-pool6-02bca4df/`
- `tmp/cuda-backend/combined-stream-pool6-02bca4df/`

## Remaining Gaps

- The host-schedule launch-overhead claim still needs graph-replay sweeps with
  distribution statistics across selected vector and tensor shapes.
- The broader paper-readiness goal still needs MPK, VDCores, serving, and
  tensor-core baseline sweeps before paper claims can be made.
