# CI Policy

## Current Mode

This standalone pto-cu repository keeps GitHub Actions disabled at the
repository settings level while the NVIDIA backend ultimate goal is active.
No runnable workflow YAML should be committed under `.github/workflows/` during
the goal. If a workflow file is reintroduced there later, it must stay
manual-only and must not register automatic `push`, `pull_request`,
`pull_request_target`, `merge_group`, or `schedule` triggers.

The purpose is to keep exploratory NVIDIA backend slices moving without being
blocked by inherited Ascend a2a3/a5 CI or repository-level check status.
Local verification, dispatch-log evidence, and changelog reports are the
required gates before a slice is pushed for human review.

## Workflow

No active workflow is currently kept under `.github/workflows/`. The archived
recipe at `docs/ci/nvidia-manual-review.workflow.yml`, named
`NVIDIA Manual Review`, records the future manual review job but is intentionally
outside GitHub's workflow directory. After Actions are explicitly reopened, a
reviewer may copy or move that recipe under `.github/workflows/` to make it
available through `workflow_dispatch` for explicit reviewer use. The recipe runs
the lightweight NVIDIA review checks:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

python -m py_compile \
  .agents/checks/check_nvidia_review_ready.py \
  examples/cuda/host_schedule_vector_ops.py \
  examples/cuda/persistent_layered_cross.py
```

## Required Local Gate

Before pushing an NVIDIA backend slice, run the cheapest relevant local checks
and record them in the dispatch log. For review-artifact changes, the default
gate is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

git diff --check
```

CUDA runtime, benchmark, or remote-evaluation slices must also run the matching
commands from `.agents/skills/cuda-backend-eval/SKILL.md` and record any
hardware or toolchain skips explicitly.

## Guardrail

The NVIDIA review guard fails if runnable workflow YAML exists under
`.github/workflows/` while this closed-CI policy is active. It also checks that
the archived review recipe stays available under `docs/ci/`. This makes closed
CI part of the branch contract instead of a temporary convention.

## Reopening CI

Repository GitHub Actions and automatic CI may be reopened only after the
NVIDIA backend reaches a stable review boundary and the workflow jobs match
this standalone repository. That change must include:

- a changelog report under `docs/nvidia-backend/changelog/`;
- an update to this document;
- an update to `.agents/checks/check_nvidia_review_ready.py`;
- human-readable evidence that the new jobs do not require inherited upstream
  a2a3/a5 infrastructure unless the branch deliberately opts into it.

If a reviewer needs to run the manual workflow before automatic CI is reopened,
temporarily copy `docs/ci/nvidia-manual-review.workflow.yml` into
`.github/workflows/`, enable repository GitHub Actions, dispatch
`NVIDIA Manual Review`, then disable Actions again, remove the runnable
workflow file, and record the run in
`docs/in_progress/nvidia_backend_paper_ready/dispatch_log.md`.
