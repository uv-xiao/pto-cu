# 2026-06-04 Plan History Compact Currentness

## Code And Data Changed

- Added `latest_reviewed_commit` to `plan_history.json` so the archive states
  which checkout boundary it summarizes.
- Updated the recent work-focus archive through `e91842e1`, including the
  model-equivalent Qwen RMSNorm mode and activation-row sample evidence.
- Limited plan-history `recent_slices` to a compact review window instead of
  letting the archive grow into a row-by-row changelog.
- Moved plan-history compactness/currentness checks into the benchmark-viewer
  validator and removed duplicate plan-history render-string assertions from
  the broad viewer artifact smoke.

## Architecture Quality

The viewer remains a short human review aid instead of a second changelog.
The validator now allows the archive to be either current or one commit behind
the checkout, which makes it useful during the commit that updates the archive
without letting several backend changes accumulate invisibly.

The test contraction follows the current testing policy: one broad viewer
smoke confirms the data file exists, while validator-level tests own the
plan-history semantics. This avoids adding more sparse UI string checks to the
largest artifact test.

## Evaluation Run

Red check before the validator update:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py \
  -q
```

Focused selectors: `test_plan_history_validator_requires_recent_checkout_commit`
and `test_plan_history_validator_rejects_verbose_recent_slices`.

Result: failed because `validate_plan_history()` did not accept the new
`allowed_latest_commits` guard parameter.

Post-fix focused check:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py \
  -q
```

Focused selectors: `test_plan_history_validator_requires_recent_checkout_commit`
and `test_plan_history_validator_rejects_verbose_recent_slices`.

Result: the focused plan-history validator tests passed.

## Remaining Gaps

This improves progress visibility and test maintainability. It does not close
the Qwen full-serving correctness gap. The next backend slice should run a
deeper model-equivalent Qwen benchmark probe before adding more reporting-only
contracts.
