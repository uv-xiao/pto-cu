# 2026-06-03 PTO Readiness Probe

## Code And Data Changed

- Added `pto_persistent_device` coverage to the paper-baseline probe guard.
- Added repo-owned path resolution for PTO probe and run-readiness entrypoint
  checks under `examples/` and `.agents/`.
- Added the committed PTO source-entrypoint probe and regenerated
  `paper_baseline_run_readiness.json` so the PTO full-serving run readiness no
  longer reports a false missing `examples/cuda/qwen_decode_loop_runner.py`.

## Architecture Quality

The paper-baseline readiness path now distinguishes external paper baselines
that run from tmp source checkouts from the standalone pto-cu target method,
whose runner and strict importer are repo-owned entrypoints. This keeps PTO
full-serving blockers focused on model-equivalent correctness and measured
raw artifacts instead of source-probe bookkeeping.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_run_readiness_uses_repo_owned_entrypoints \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_paper_baseline_probe_covers_repo_owned_entrypoints \
  -q
```

The first focused test failed while the PTO runner path was marked `fail`; the
second failed because no `pto_persistent_device` probe existed.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_run_readiness_uses_repo_owned_entrypoints \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_paper_baseline_probe_covers_repo_owned_entrypoints \
  -q
```

Result: both PTO readiness selectors pass after regenerating the probe and
run-readiness artifacts.

## Remaining Gaps

PTO run readiness is now source-ready, but the paper queue still requires
model-equivalent MPK-policy and VDCores-policy Qwen3-8B full-serving rows with
latency and throughput metrics before importing final paper-grade evidence.
