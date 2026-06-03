# 2026-06-03 Baseline Readiness Paths

## Code And Data Changed

- Fixed paper-baseline run-readiness entrypoint checks so commands that name a
  repo-relative `tmp/` source path are resolved from the repository root
  instead of being appended to the baseline source checkout twice.
- Replaced the stale ThunderKittens `<selected-kernel>` placeholder in the
  imported `thunderkittens_tile_kernel` run with the selected
  `kernels/attention/mha_h100` correctness and benchmark scripts.
- Regenerated review artifacts so MPK native-vs-persistent and
  ThunderKittens tile-kernel run readiness now report `pass`.

## Architecture Quality

Run readiness now distinguishes concrete reproduction paths from local source
checkout-relative paths. This keeps the paper queue focused on missing measured
full-serving evidence rather than false source-path blockers in already
surveyed baseline commands.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_mpk_run_readiness_accepts_repo_relative_tmp_entrypoint \
  tests/ut/py/test_nvidia_review_artifacts.py::test_thunderkittens_tile_readiness_uses_selected_mha_kernel \
  -q
```

The MPK test failed while `tmp/baselines/mirage-mpk/demo/qwen3/demo.py` was
reported missing, and the ThunderKittens test failed while readiness still
emitted generic `test_correctness.py` checks.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_mpk_run_readiness_accepts_repo_relative_tmp_entrypoint \
  tests/ut/py/test_nvidia_review_artifacts.py::test_thunderkittens_tile_readiness_uses_selected_mha_kernel \
  -q
```

Result: both focused readiness checks pass after regenerating review artifacts.

## Remaining Gaps

This removes stale readiness blockers only. Final paper-grade status still
requires importing the queued PTO, VDCores, and ThunderKittens full-serving
evidence and closing the backend implementation gaps tracked in status docs.
