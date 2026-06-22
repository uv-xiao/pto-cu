# Restart PR Slicing Plan

This plan tracks the current PR-sized route for continuing the NVIDIA backend
restart. The dirty orphan workspace is historical input only; new work starts
from `main` and lands through focused GitHub PRs.

## Current Baseline

- Base branch: `main`.
- Repository hygiene PRs have already moved agent guidance to `.agents/`,
  added interval-based Codex goal monitoring, and merged the latest
  FlashAttention append coverage slice.
- The dispatcher branch `restart-tracking-docs` owns only restart tracking
  restoration under `docs/in_progress/`.

## Rules

- Each PR must stage only its owned paths. It must not bundle unrelated dirty
  worktree state.
- Use a fresh branch from current `main` for every child slice.
- Keep `docs/in_progress/nvidia_backend/dispatch_log.md` current for every
  branch, verification command, PR, merge decision, and handoff.
- Use interval-based monitor summaries before opening full tmux or transcript
  captures.
- Do not promote `docs/in_progress/` into `docs/nvidia-backend/` until the
  corresponding PR has been reviewed and accepted.

## Completed Bootstrap Slices

- PR #137: Codex goal monitor recurring tick support.
  - Result: merged as `7ab6dad31a6641c160305299a72467c74689486a`.
- PR #138: consolidate agent guidance under `.agents`.
  - Result: merged as `9e793ab1f51ccf2ffbf5120005acc2cddf42b843`.
- PR #136: broader Gluon FlashAttention append sweep coverage.
  - Result: merged as `c3964aad9204d0fbd0042ebbd7f88309530b80d2`.

## Active PR - Restart Tracking Restoration

Objective: restore a concise, current umbrella goal, source/skill notes,
artifact audit, slicing plan, and dispatch log so future workers have durable
context without committing a scrollback-sized historical log.

PR: <https://github.com/uv-xiao/pto-cu/pull/139>.

Owned paths:

- `docs/in_progress/001-nvidia-backend.md`
- `docs/in_progress/nvidia_backend/source_manifest.md`
- `docs/in_progress/nvidia_backend/skill_selection.md`
- `docs/in_progress/nvidia_backend/cuda_eval_script_selection.md`
- `docs/in_progress/nvidia_backend/current_cuda_artifact_audit.md`
- `docs/in_progress/nvidia_backend/pr_slicing_plan.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`

Verification gate:

```bash
git diff --check
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

Non-claims:

- This PR does not implement new CUDA runtime behavior.
- This PR does not run new H200 kernels.
- This PR does not prove new serving or DeepSeek output behavior.

## Next Candidate Slices

Choose the next child slice after this tracking PR is reviewed:

1. Audit accepted `main` against the umbrella acceptance criteria and identify
   the smallest missing requirement that needs implementation rather than
   documentation.
2. Continue distributed communication work if current evidence still lacks the
   required multi-GPU MoE dispatch/combine path.
3. Continue serving promotion if current vLLM/DeepSeek evidence is ready to
   move from in-progress notes into stable review docs.
4. Retire redundant in-progress docs only after their evidence is represented
   in accepted stable docs or no longer needed.
