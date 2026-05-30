# 2026-05-31 NVIDIA Branch CI

## Code And Data Changed

- Replaced the branch CI workflow with a focused NVIDIA review job.
- Closed the repository base-branch PR trigger for the NVIDIA ultimate goal by
  keeping the workflow available through manual `workflow_dispatch` only.
- Removed Ascend sim and hardware jobs from this branch workflow so the NVIDIA
  backend PR does not schedule a2a3/a5 CI.
- Added a review-artifact test and NVIDIA review guard coverage that scan
  every workflow file for automatic triggers.
- Replaced the stale inherited CI documentation with the standalone pto-cu
  manual-CI policy for the NVIDIA ultimate goal.

## Architecture Quality

The branch CI is scoped to NVIDIA review evidence and avoids Ascend-specific
hardware queues while this standalone branch is in ultimate-goal mode. The
guard now treats manual-only CI as a repository-wide workflow invariant, not
just a property of the current `ci.yml` file.

The manual review workflow runs:

- `.agents/checks/check_nvidia_review_ready.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`
- Python syntax checks for the review guard and CUDA example wrappers

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

## Remaining Gaps

- Automatic GitHub Actions remain closed during the ultimate goal. Reviewers
  should rely on local guard/test evidence or manually dispatch the workflow.
- Any future change that reopens automatic CI must update `docs/ci.md`, the
  review guard, and a changelog report in the same slice.
