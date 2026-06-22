# NVIDIA Backend Dispatch Log

This log records dispatcher-worker activity for
`docs/in_progress/001-nvidia-backend.md`. It is required review evidence; do
not rely on private terminal scrollback, unstated session memory, or unmerged
local changes.

## Logging Schema

Each dispatch entry should include:

- timestamp;
- dispatcher session or PR;
- worker id and objective;
- exact Codex command or script invocation;
- parent goal and child slice;
- branch name and PR URL or planned PR slot;
- allowed scope and files;
- dependencies and blocked assumptions;
- verification commands and results;
- merge decision and merge commit, when applicable;
- handoff summary and remaining gaps.

## Entries

### 2026-06-22 - Restart Tracking Restoration Branch

- Dispatcher Session or PR:
  current local goal session following the recovered restart objective.
- Worker id and objective:
  dispatcher-owned restart tracking restoration; restore durable tracking docs
  without committing the historical 16k-line dispatch scrollback.
- Exact Codex command or script invocation:
  direct dispatcher documentation branch `restart-tracking-docs`.
- Parent goal and child slice:
  NVIDIA backend restart; dispatcher artifacts and repository hygiene.
- Branch name and PR URL or planned PR slot:
  `restart-tracking-docs`;
  <https://github.com/uv-xiao/pto-cu/pull/139>.
- Allowed scope and files:
  `docs/in_progress/001-nvidia-backend.md` and focused
  `docs/in_progress/nvidia_backend/` tracking notes.
- Dependencies and blocked assumptions:
  the recovered backup snapshot remains objective context, but current
  tracking docs must not require a root restart file to exist.
- Verification commands and results:
  `git diff --check`, `git diff --cached --check`, targeted
  `markdownlint-cli2` over the seven changed docs, NVIDIA review guard, and
  focused review-artifact tests (`58 passed`) passed.
- Merge decision and merge commit:
  exact-head squash merge succeeded. PR #139 merged as
  `3722ad7efd7257fcc3807111aa449bfb49c57ea3`.
- Handoff summary and remaining gaps:
  restored tracking docs are now on `main`. The dispatcher should next select
  an implementation or evidence gap from the umbrella acceptance criteria and
  launch a PR-sized worker from current `main`.

### 2026-06-22 - Merge Dispatcher Hygiene PRs

- Dispatcher Session or PR:
  current local goal session following the recovered restart objective.
- Worker id and objective:
  dispatcher-owned repository hygiene; restore credible Codex operating
  surface before continuing feature work.
- Exact Codex command or script invocation:
  direct dispatcher PR management with GitHub CLI and exact-head REST merges.
- Parent goal and child slice:
  repository organization and long-running agent workflow.
- Branch name and PR URL or planned PR slot:
  `codex-goal-monitor-scheduler`;
  <https://github.com/uv-xiao/pto-cu/pull/137>.
  `codex-agent-guidance-cleanup`;
  <https://github.com/uv-xiao/pto-cu/pull/138>.
- Allowed scope and files:
  `.agents/skills/codex-goal-monitor/`, `AGENTS.md`, `.agents/**`,
  `README.md`, and docs that referenced retired guidance locations.
- Dependencies and blocked assumptions:
  PR #137 restored missing recurring scheduler/tick behavior for the Codex
  adaptation of upstream `monitor-codex-goal`. PR #138 removed retired root
  guidance and moved reusable always-on rules under `.agents/rules/`.
- Verification commands and results:
  PR #137: shell syntax check for
  `monitor-codex-goal-tick.sh`, self-test tick with missing pane/transcript,
  `git diff --check`, and `git diff --cached --check` passed. PR #138: stale
  reference scans, no tracked retired-guidance paths, diff checks, NVIDIA
  review guard, and focused review-artifact tests passed.
- Merge decision and merge commit:
  exact-head squash merges succeeded. PR #137 merged as
  `7ab6dad31a6641c160305299a72467c74689486a`; PR #138 merged as
  `9e793ab1f51ccf2ffbf5120005acc2cddf42b843`.
- Handoff summary and remaining gaps:
  root checkout was fast-forwarded to current `origin/main` and remained clean.
  Remaining restart work still requires tracking restoration, further
  distributed CUDA slices, serving promotion, and DeepSeek evidence.

### 2026-06-22 - Merge Gluon FlashAttention Append Coverage

- Dispatcher Session or PR:
  current local goal session after PR #135 merged.
- Worker id and objective:
  `pto-worker-gluon-flashattention-append-coverage-h200`; broaden bounded
  causal append sweep coverage for the existing Gluon FlashAttention harness.
- Exact Codex command or script invocation:
  child Codex worker launched in tmux with a branch-specific prompt under
  `tmp/worker-prompts/`.
- Monitor locators:
  tmux pane and transcript were recorded in the historical dispatch log; future
  monitoring should use interval-based tick summaries instead of repeatedly
  capturing full tmux/transcript context.
- Parent goal and child slice:
  attention evidence gap after PR #135; this child owned only broader bounded
  causal append sweep coverage.
- Branch name and PR URL or planned PR slot:
  `gluon-flashattention-append-coverage-h200`;
  <https://github.com/uv-xiao/pto-cu/pull/136>.
- Allowed scope and files:
  FlashAttention example, focused compiler tests, focused review-artifact
  tests, CUDA example README, and FlashAttention/checklist evidence docs.
- Dependencies and blocked assumptions:
  worker had to preserve existing non-causal, causal prefill, causal decode,
  current causal append, single-case, and unsupported-boundary evidence. It
  could not claim full append coverage, serving integration, DeepSeek semantic
  correctness, production readiness, or performance evidence.
- Verification commands and results:
  child PR #136 opened at `99c8e077df29f2effd94823dcfd94d55464a82af` after
  passing focused compiler tests, focused review-artifact tests, NVIDIA review
  guard, Markdown lint on changed docs, diff checks, private-path scans, and
  H200 causal append aggregate sweep evidence. Parent rechecked the merge
  candidate against updated `origin/main`, reran focused tests and NVIDIA
  guard, and reran the H200 causal append aggregate sweep through
  `run-remote-cuda.sh --sync`; result was `status: passed`, `case_count: 3`,
  `passed_cases: 3`, all cases `phase: append` and `causal: true`, on NVIDIA
  H200 NVL GPUs with driver `580.126.20` and CUDA toolkit `12.8`.
- Merge decision and merge commit:
  exact-head squash merge against
  `99c8e077df29f2effd94823dcfd94d55464a82af` succeeded as
  `c3964aad9204d0fbd0042ebbd7f88309530b80d2`; the remote worker branch was
  deleted.
- Handoff summary and remaining gaps:
  slice produced broader bounded causal append sweep evidence only. It is not
  paged/ragged KV-cache correctness, varlen correctness, attention-variant
  correctness, full prefill, full decode, full append, serving, performance,
  or DeepSeek semantic evidence.
