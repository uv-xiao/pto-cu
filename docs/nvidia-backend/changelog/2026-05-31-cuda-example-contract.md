# 2026-05-31 CUDA Example Contract

## Code And Data Changed

- Added `examples/cuda/manifest.json`.
- Added `.agents/checks/validate_cuda_examples.py`.
- Wired the CUDA example validator into the NVIDIA review guard.
- Updated `examples/cuda/README.md` with benchmark IDs, runtime IDs, method
  IDs, commands, and expected output for each CUDA example.
- Extended focused review artifact tests to run the example validator.

## Architecture Quality

CUDA examples now have a machine-checkable link to the benchmark viewer data.
Each example declares the benchmark it demonstrates, the runtime/method it
exercises, its command, expected output, and script-level evidence symbols.

This keeps examples from drifting into unreviewed wrappers that no longer map
to evaluated workloads.

## Evaluation Run

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_cuda_examples.py

git diff --check
```

## Remaining Gaps

- The current examples cover host-schedule vector ops and the persistent
  layered-cross graph. Future paper slices should add examples for tensor-core
  sweeps, stream/graph comparisons, and imported MPK/VDCores baselines when
  those workflows become runnable in this repo.
