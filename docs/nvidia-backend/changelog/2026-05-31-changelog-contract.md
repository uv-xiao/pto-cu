# 2026-05-31 Changelog Contract

## Code And Data Changed

- Added `.agents/checks/validate_nvidia_changelog.py`.
- Wired the changelog validator into the NVIDIA review guard.
- Normalized earlier changelog reports to the required review sections.
- Extended focused review artifact tests to run the changelog validator.

## Architecture Quality

Changelog reports are now machine-checkable review artifacts. The guard checks
that every report is linked from the changelog index and that every report has
sections for changed code/data, architecture quality, evaluation evidence, and
remaining gaps.

This makes the reporting workflow durable across child PRs instead of relying
on reviewer memory or informal report shape.

## Evaluation Run

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_nvidia_changelog.py

git diff --check
```

## Remaining Gaps

- The validator checks report structure and index coverage. It does not prove
  that the described runtime behavior is implemented; that remains covered by
  evidence refs, viewer-data validation, tests, and future runtime evaluation.
