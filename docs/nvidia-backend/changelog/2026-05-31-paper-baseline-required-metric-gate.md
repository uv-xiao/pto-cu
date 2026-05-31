# 2026-05-31 Paper Baseline Required Metric Gate

## Code And Data Changed

- Added required-metric validation to `paper_baseline_results_update.py`.
- The updater now rejects raw paper-baseline rows that do not satisfy the
  referenced `paper_baseline_runs.json` `required_metrics` list.
- Added a focused regression test proving an incomplete MPK scheduler trace
  cannot write viewer rows, mark a run imported, or regenerate the audit.
- Updated the CUDA evaluation workflow and shared contracts to describe the
  updater as the measured-evidence acceptance gate.

## Architecture Quality

The paper-baseline run contract is now enforced at import time. Future MPK,
VDCores, vLLM, SGLang, and ThunderKittens artifacts cannot become committed
viewer evidence unless their raw JSON includes the metrics the paper plan says
are required. This keeps documentation, planned run contracts, and committed
viewer data aligned.

## Evaluation Run

The guard was developed with a failing fixture first:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_results_update_rejects_missing_required_metric'
```

The existing successful update fixture still passes for a complete MPK
scheduler trace, proving the new check rejects incomplete data without blocking
valid measured rows.

## Remaining Gaps

This is an acceptance guard, not a new benchmark result. The project still
needs measured MPK/VDCores scheduler imports, MPK/VDCores/vLLM/SGLang serving
captures, and broader ThunderKittens correctness/benchmark sweeps before the
blocked paper claims can be promoted.
