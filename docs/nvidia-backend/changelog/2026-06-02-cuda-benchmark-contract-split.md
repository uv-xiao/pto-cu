# CUDA Benchmark Contract Split

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark_impl/contracts.py`
  for source-paper metadata and report baseline categories.
- Updated `cuda_benchmark.py` to import those contracts while preserving the
  same script-level names used by report rendering and tests.

## Architecture Quality

The host-schedule benchmark harness now keeps review-facing evaluation
contracts outside the 4k-line runner. This makes the MPK/VDCores paper-source
metadata and tensor-throughput baseline groups easier to inspect without
touching benchmark execution paths.

## Evaluation Run

- `py_compile` covered the benchmark entrypoint and new contracts module.
- Focused pytest covered source-paper metadata, payload merge metadata, and
  tensor-throughput report rendering: `3 passed, 321 deselected`.

## Remaining Gaps

`cuda_benchmark.py` is still oversized. Driver/runtime wrappers, sample
execution, and report rendering remain candidates for later module splits.
