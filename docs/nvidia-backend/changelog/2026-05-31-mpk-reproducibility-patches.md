# 2026-05-31 MPK Reproducibility Patches

## Code And Data Changed

- Added committed MPK baseline patch files under
  `docs/nvidia-backend/baseline-patches/`:
  - `mpk-snapshot-pointer-runtime-config.patch`
  - `mpk-predecode-token-dump.patch`
- Added `reproducibility_patches` to the patched MPK execution attempts in
  `paper_baseline_execution_attempts.json`.
- Extended the benchmark-viewer validator so any execution attempt that
  depends on a local baseline patch must name committed `.patch` files.
- Updated the review-artifact test to read the patch files and require the
  snapshot pointer and predecode token-dump evidence.

## Architecture Quality

The MPK diagnostic path no longer depends only on mutable ignored files under
`tmp/baselines/mirage-mpk`. The raw execution artifacts still live under
`tmp/`, but the patch needed to reproduce the baseline behavior is now a
reviewable repository artifact.

This keeps upstream read-only while making the baseline delta explicit. The
snapshot-pointer patch is the candidate reproducibility fix. The predecode
patch is diagnostic-only and exists so reviewers can reproduce the token-state
capture that found generated token id `-1`.

## Evaluation Run

The focused TDD check first failed because the patched MPK execution attempt
records had no `reproducibility_patches` field. After adding patch files and
viewer data, the focused check passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'benchmark_viewer_data_contracts_are_complete or benchmark_viewer_schema_validator_passes'
```

The standalone viewer-data validator also passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

- The patch files make the local MPK baseline delta reproducible, but they do
  not make MPK paper-grade by themselves.
- MPK still needs valid generated token IDs under sanitizer, token export,
  scheduler/resource-policy metrics, and latency rows before its persistent
  scheduler result can be imported as paper-ready evidence.
