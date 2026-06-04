# 2026-06-04 Plan History HEAD Currentness

## Code And Data Changed

- Tightened the benchmark-viewer plan-history validator so
  `latest_reviewed_commit` must match the current checkout by default, with a
  bounded parent-commit exception for plan-history maintenance commits.
- Kept unit tests able to inject an explicit allowed commit set, so focused
  validator tests do not depend on the repository's live `HEAD`.
- Refreshed `plan_history.json` through `e7a08c16`, including the residual
  BF16 boundary slice and a compact eight-entry recent-slice window.

## Architecture Quality

The plan archive is meant to be a short review aid, not a second changelog.
Allowing `HEAD^` unconditionally let the archive pass while one runtime slice
behind the current branch, which weakened the user's requested reflection loop.
A checked-in JSON file cannot name the hash of the commit that contains its own
content, so the validator keeps a narrow `HEAD^` escape hatch only when the
current commit changed plan-history maintenance files.

This is intentionally a small guardrail update. It does not add artifact-level
row tests, new viewer fields, or another broad UI assertion. The follow-up work
should return to benchmark-model correctness unless the archive itself is
stale or invalid.

## Evaluation Run

Red check:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/.agents/checks .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py::test_plan_history_validator_accepts_parent_for_archive_maintenance_commit \
  -q
```

Result: failed because the validator did not yet recognize the bounded
plan-history maintenance exception.

Post-fix checks cover the focused selectors, the benchmark-viewer data contract,
and the NVIDIA review guard.

## Remaining Gaps

This makes the review archive current through the residual BF16 boundary
commit. It does not close Qwen full-serving correctness; the next feature slice
should stay on the layer-3 top-k numeric root cause.
