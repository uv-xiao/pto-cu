# NVIDIA Backend Paper-Ready Work Preparation

## Read Order

1. `CLAUDE.md`
2. `.claude/rules/`
3. `.agents/rules/ultimate-goal-dispatch.md`
4. `docs/in_progress/nvidia_backend_paper_ready.md`
5. `docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md`
6. `docs/in_progress/nvidia_backend_paper_ready/evaluation_plan.md`
7. `docs/nvidia-backend/overall.md`
8. `docs/nvidia-backend/flows.md`
9. `docs/nvidia-backend/persistent-device.md`
10. `docs/nvidia-backend/evaluation.md`
11. `docs/nvidia-backend/status.md`

## Repository Policy

This is a standalone pto-cu project. Branches, pushes, and PRs for this goal
target `origin` and `uv-xiao/pto-cu:main`. Do not write to upstream
repositories, do not edit upstream PRs, and do not depend on upstream branch
settings for this work.

## Branch And PR Policy

- Use one child branch per reviewable slice.
- Keep child PRs small enough for a human to review code, docs, evidence, and
  verification together.
- During this ultimate goal, GitHub Actions are manual-only so repository CI
  does not block exploratory child slices. Every workflow under
  `.github/workflows/` must avoid automatic triggers. Local verification and
  dispatch-log evidence are the required gates before pushing a slice.
- Update the dispatch log before launching a worker and after reviewing its
  result.
- A child PR is not ready until code, docs, examples, viewer data, and
  changelog reports agree.

## Remote Evaluation Fallback

Remote evaluation has two allowed refresh paths:

- Git path: fetch and checkout the requested pto-cu branch on the remote host.
- Tree-sync fallback: copy the local checkout to the remote host through SSH
  when remote Git credentials, network, or HTTPS transport fail.

Any remote run must record which path was used, the local commit, remote commit
or tree-sync source commit, remote directory, CUDA toolkit path, GPU model,
command, output JSON path, and whether raw artifacts were copied back under
`tmp/`.

## Source Notes

Put used external sources under `tmp/`, including paper PDFs, extracted text,
cloned baseline repositories, command logs, and short source indexes. Stable
committed docs should cite the source name and local tmp path, but should not
commit raw downloaded sources.

## Verification Policy

Use the cheapest relevant checks for each child slice, then broaden when the
slice changes runtime behavior:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

git diff --check
```

CUDA runtime slices must also run the matching smoke or benchmark commands from
`.agents/skills/cuda-backend-eval/SKILL.md`. Record skipped commands with the
reason and the missing hardware, driver, or toolchain dependency.

## Worker Rules

Workers own one child slice and must not dispatch nested workers. If a worker
finds more parallel work, it records a proposed child slice in the dispatch
log or a handoff file named by the dispatcher.
