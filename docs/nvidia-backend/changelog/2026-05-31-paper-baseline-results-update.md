# 2026-05-31 Paper Baseline Results Update

## Code And Data Changed

- Added `paper_baseline_results_update.py` under the CUDA backend evaluation
  scripts.
- Added a focused test that imports an MPK scheduler raw JSON fixture, writes
  viewer records, updates a copied `results.json`, marks
  `mpk_persistent_scheduler_trace` as `imported_to_viewer`, and regenerates a
  copied paper-readiness audit.
- Updated the evaluation skill, shared contracts, and paper-ready evaluation
  plan to route committed paper-baseline rows through the updater.

## Architecture Quality

The paper-baseline path now has a single reviewable handoff from measured raw
artifact to committed viewer state. Future MPK, VDCores, vLLM, SGLang, or
ThunderKittens imports should update `results.json`, `paper_baseline_runs.json`,
and `paper_readiness_audit.json` together instead of by separate hand edits.

## Evaluation Run

The updater fixture was developed red/green with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_results_update_marks_imported_run'
```

The initial run failed because the updater script was missing. After adding
the script, the fixture passed.

## Remaining Gaps

This change does not add new measured MPK or VDCores results. Those baselines
must still be built, run, captured under `tmp/cuda-backend/paper-baselines/`,
then imported through the updater before the paper-readiness blockers can be
removed.
