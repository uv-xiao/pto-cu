# 2026-05-31 Direct Launch Sweep Evidence

## Code And Data Changed

- Added `cuda_direct_launch_sweep.py`, a paired A100/H200 runner for selected
  CUDA Runtime and CUDA Driver direct-launch vector and tensor shapes.
- Added focused tests for command construction, remote tree-sync source
  stamping, dry-run metadata, source-paper provenance, and report generation.
- Extended benchmark-viewer capture import rules for direct Runtime and Driver
  vector rows and naive SGEMM rows at `1024`, `4096`, and `65536`.
- Imported 24 direct-launch sweep result records into the benchmark viewer.
- Updated the paper evaluation matrix and audit so the host-schedule
  launch-overhead claim is the first claim marked ready for paper review.

## Architecture Quality

The runner mirrors the graph-replay sweep runner but deliberately limits its
baseline set to `direct_runtime`, `direct_driver`, `direct_runtime_sgemm`, and
`direct_driver_sgemm`. That keeps the evidence slice aligned with the
remaining host-schedule blocker: direct launches over the same selected vector
and tensor shapes already used for graph replay.

The remote path uses tree sync plus `PTO_SOURCE_COMMIT`, so H200 evaluation
does not depend on remote Git access. The generated capture still records the
source commit and command examples for human review.

## Evaluation Run

The TDD red check first failed because the direct-launch sweep runner did not
exist:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_direct_launch_sweep_builds_a100_h200_workflow \
  tests/ut/py/test_cuda_benchmark_report.py::test_cuda_direct_launch_sweep_dry_run_records_source_papers \
  -q
```

After adding the runner, the same focused test passed with `2 passed`.

The paired capture command was:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_direct_launch_sweep.py \
  --branch goal/nvidia-paper-ready --sizes 1024,4096,65536 --repeats 10 \
  --sync-remote-tree
```

The validator passed for 240 rows, machines `hina` and `dasys-h200x8`,
baselines `direct_runtime`, `direct_driver`, `direct_runtime_sgemm`, and
`direct_driver_sgemm`, tensor tile `16x16x16`, source-paper provenance,
command examples, and report files.

Raw artifact:

- `tmp/cuda-backend/direct-launch-sweep-626b8c75/`

## Remaining Gaps

- The broader paper-readiness goal still needs MPK, VDCores, serving, and
  tensor-core baseline sweeps before the project can claim full paper-ready
  evaluation coverage.
