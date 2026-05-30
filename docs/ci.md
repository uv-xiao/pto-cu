# CI Policy

## Current Mode

This standalone pto-cu repository keeps GitHub Actions manual-only while the
NVIDIA backend ultimate goal is active. Workflow files under
`.github/workflows/` must not register automatic `push`, `pull_request`,
`pull_request_target`, `merge_group`, or `schedule` triggers.

The purpose is to keep exploratory NVIDIA backend slices moving without being
blocked by inherited Ascend a2a3/a5 CI. Local verification, dispatch-log
evidence, and changelog reports are the required gates before a slice is pushed
for human review.

## Workflow

The only workflow currently kept in the repository is
`.github/workflows/ci.yml`, named `NVIDIA Manual Review`. It is available
through `workflow_dispatch` for explicit reviewer use and runs the lightweight
NVIDIA review checks:

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

The NVIDIA review guard scans every file in `.github/workflows/` and fails if a
workflow reintroduces automatic triggers or Ascend-specific runner usage. This
makes the manual-only CI policy part of the branch contract instead of a
temporary convention.

## Reopening CI

Automatic CI may be reopened only after the NVIDIA backend reaches a stable
review boundary and the workflow jobs match this standalone repository. That
change must include:

- a changelog report under `docs/nvidia-backend/changelog/`;
- an update to this document;
- an update to `.agents/checks/check_nvidia_review_ready.py`;
- human-readable evidence that the new jobs do not require inherited upstream
  a2a3/a5 infrastructure unless the branch deliberately opts into it.
