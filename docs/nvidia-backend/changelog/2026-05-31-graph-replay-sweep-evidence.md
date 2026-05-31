# 2026-05-31 Graph Replay Sweep Evidence

## Code And Data Changed

- Added `cuda_graph_replay_sweep.py`, a paired A100/H200 runner for selected
  CUDA Driver graph replay vector and tensor launch shapes.
- Added focused tests for graph-replay command construction, remote tree-sync
  source-commit stamping, dry-run metadata, and report generation.
- Extended benchmark-viewer capture import rules for graph replay sizes
  `1024`, `4096`, and `65536`.
- Imported 12 graph-replay sweep result records into the benchmark viewer.
- Updated the paper evaluation matrix and audit to replace the graph-replay
  blocker with the remaining direct runtime/direct driver sweep blocker.

## Architecture Quality

The runner is intentionally narrower than the full paired benchmark runner:
it captures only `direct_driver_graph` and `direct_driver_graph_sgemm`, which
keeps the artifact aligned to the host-schedule launch-overhead blocker being
closed. It uses the existing `cuda_benchmark.py` single-baseline path and
standard `cuda-benchmark.*` report renderer, so viewer import and validation
reuse the same contracts as prior raw captures.

The remote path uses tree sync plus `PTO_SOURCE_COMMIT`, avoiding remote Git
fetch as the source of truth while still preserving the evaluated source
commit in the H200 sample commands.

## Evaluation Run

The TDD red check first failed because the graph-replay sweep runner did not
exist. After adding the runner, the focused tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_graph_replay_sweep_builds_a100_h200_workflow \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_graph_replay_sweep_dry_run_records_source_papers \
  -q
```

Result after the machine-stamping fix: `2 passed in 1.42s`.

The paired capture command was:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_graph_replay_sweep.py \
  --branch goal/nvidia-paper-ready --sizes 1024,4096,65536 --repeats 10 \
  --sync-remote-tree
```

The validator passed for 120 rows, machines `hina` and `dasys-h200x8`,
baselines `direct_driver_graph` and `direct_driver_graph_sgemm`, tensor tile
`16x16x16`, source-paper provenance, command examples, and report files.

Raw artifact:

- `tmp/cuda-backend/graph-replay-sweep-01e30e99/`

## Remaining Gaps

- The host-schedule launch-overhead claim still needs direct runtime and
  direct driver sweeps across the same selected vector and tensor shapes.
- The broader paper-readiness goal still needs MPK, VDCores, serving, and
  tensor-core baseline sweeps before paper claims can be made.
