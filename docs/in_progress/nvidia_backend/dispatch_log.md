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

### 2026-06-22 - Post-Fused-Boundary Status Refresh Worker

- Dispatcher Session or PR:
  child worker session after PR #143 merged.
- Worker id and objective:
  `pto-worker-nvidia-goal-status-post-fused-boundary`; refresh the NVIDIA
  backend restart status surface on current `main` after PR #143 and open one
  docs/status PR.
- Exact Codex command or script invocation:
  child `/goal` worker in worktree
  `nvidia-goal-status-post-fused-boundary`.
- Parent goal and child slice:
  NVIDIA backend restart; post-#143 status audit and next-slice selection
  only.
- Branch name and PR URL or planned PR slot:
  `nvidia-goal-status-post-fused-boundary`;
  <https://github.com/uv-xiao/pto-cu/pull/144>.
- Allowed scope and files:
  `docs/in_progress/nvidia_backend/goal_status_rollup.md`,
  `docs/in_progress/nvidia_backend/pr_slicing_plan.md`,
  `docs/in_progress/nvidia_backend/dispatch_log.md`, and focused guard
  updates in `tests/ut/py/test_nvidia_review_artifacts.py`.
- Dependencies and blocked assumptions:
  starts from current `main` at
  `f73620c613b7a97c352384d6e90f32ae8c4106cd`, where PR #143 records
  `--with-uccl-ep-fused-boundary` as `status: unsupported` because
  `persistent_device_uccl_ep_runtime_fusion` is missing.
- Verification commands and results:
  `git diff --check` passed; `git diff --cached --check` passed; targeted
  `markdownlint-cli2` over the three touched docs passed; the NVIDIA review
  guard passed; and `test_nvidia_review_artifacts.py` passed with
  `60 passed`.
- Merge decision and merge commit:
  pending PR #144 review and merge decision.
- Handoff summary and remaining gaps:
  status docs now audit current `main` at `f73620c6`, keep #143 as structured
  unsupported-boundary evidence instead of fused execution evidence, and
  recommend `nvidia-uccl-ep-runtime-fusion-design` as the next conservative
  dependency/design slice. Remaining gaps include actual
  `persistent_device_uccl_ep_runtime_fusion`, CUDA host-runtime UCCL
  dispatch, RDMA or multi-node evidence, serving/vLLM simpler-nv integration,
  DeepSeek correctness through simpler-nv kernels, and throughput/latency
  evidence.

### 2026-06-22 - NVIDIA MoE UCCL-EP Fused Boundary Worker

- Dispatcher Session or PR:
  child worker session for the next slice selected by the goal status rollup.
- Worker id and objective:
  `pto-worker-nvidia-moe-uccl-ep-fused-boundary-h200`; convert the accepted
  persistent-MoE plus UCCL-EP adapter handoff into an explicit reduced fused
  cross-GPU expert-parallel MoE boundary result.
- Exact Codex command or script invocation:
  child `/goal` worker in worktree
  `nvidia-moe-uccl-ep-fused-boundary-h200`.
- Parent goal and child slice:
  NVIDIA backend restart; first reduced fused cross-GPU expert-parallel MoE
  boundary status from the accepted persistent MoE plus UCCL-EP handoff.
- Branch name and PR URL or planned PR slot:
  `nvidia-moe-uccl-ep-fused-boundary-h200`;
  <https://github.com/uv-xiao/pto-cu/pull/143>.
- Allowed scope and files:
  `examples/cuda/persistent_moe_dispatch_combine.py`,
  `tests/ut/py/test_cuda_comm.py`,
  `tests/ut/py/test_nvidia_review_artifacts.py`,
  `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`,
  `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`,
  `docs/in_progress/nvidia_backend/communication_selection.md`, and this
  dispatch log.
- Dependencies and blocked assumptions:
  H200 UCCL-EP execution used sanitized external dependency paths:
  `<external-uccl-ep-bench>` for the UCCL checkout and EP bench helper, and
  `<uccl-python-site-packages>` for external Torch site packages. The command
  copied the prebuilt `uccl.ep` extension into the project-local remote venv.
- Verification commands and results:
  `git diff --check` passed; `git diff --cached --check` passed; targeted
  `markdownlint-cli2` over the persistent MoE, communication boundary,
  communication selection, and dispatch log docs passed; `test_cuda_comm.py`
  passed with `33 passed`; the NVIDIA review guard passed; and
  `test_nvidia_review_artifacts.py` passed with `60 passed`.
- H200 evidence:
  remote `run-remote-cuda.sh --sync` on `NVIDIA H200 NVL` devices `6,7`,
  driver `580.126.20`, CUDA toolkit under `/usr/local/cuda`, Python `3.12.3`.
  The command exited `3` with `status: unsupported`. Persistent MoE validation
  passed on both devices with `max_abs_error: 0.0`, completed count `5`,
  zero scheduler errors, zero fan-in remaining, and matching source/bridge
  digests. UCCL-EP adapter validation passed on both ranks with
  `max_abs_error: 0.0`, `topk_weight_error: 0.0`, descriptor metadata present,
  and `recv_tokens: [88]` on both ranks. The result is non-evidence for actual
  fused cross-GPU expert-parallel MoE because
  `persistent_device_uccl_ep_runtime_fusion` remains missing.
- Merge decision and merge commit:
  accepted as a structured unsupported-boundary status slice only. PR #143
  merged into `main` on 2026-06-22 as
  `f73620c613b7a97c352384d6e90f32ae8c4106cd`
  (`Record UCCL EP fused boundary status`). The merge decision did not accept
  the result as actual fused cross-GPU expert-parallel MoE execution evidence.
- Handoff summary and remaining gaps:
  the branch adds `--with-uccl-ep-fused-boundary` and records a structured
  unsupported boundary instead of promoting the prior handoff to fused
  evidence. Remaining gaps are actual persistent-device/UCCL-EP runtime
  fusion, device-side cross-GPU expert routing, CUDA host-runtime UCCL
  dispatch, RDMA or multi-node evidence, serving/vLLM integration,
  DeepSeek correctness, and throughput/latency evidence.

### 2026-06-22 - Dispatch Goal Status Rollup Worker

- Dispatcher Session or PR:
  current local goal session after PR #140 merged.
- Worker id and objective:
  `pto-worker-nvidia-goal-status-rollup`; produce a concise current-status
  rollup mapping the NVIDIA backend umbrella acceptance criteria to accepted,
  partial, missing, or promotion-needed evidence from current `main`.
- Exact Codex command or script invocation:
  `tmux new-session -d -s pto-worker-nvidia-goal-status-rollup
  'cd <worker-worktree>/nvidia-goal-status-rollup &&
  codex -m gpt-5.5 -a never -s danger-full-access exec - <
  <dispatcher-root>/tmp/worker-prompts/nvidia-goal-status-rollup.md'`.
- Monitor locators:
  tmux pane `pto-worker-nvidia-goal-status-rollup:0.0`; transcript
  `~/.codex/sessions/2026/06/22/rollout-2026-06-22T16-11-35-019eee62-8a33-7eb1-ba4a-9eec12aad66c.jsonl`.
  Recurring monitor pane `pto-monitor-nvidia-goal-status-rollup:0.0` writes
  summary-first ticks under
  `tmp/codex-goal-monitor/nvidia-goal-status-rollup/` every 30 minutes.
- Parent goal and child slice:
  recovered restart objective; status audit and next-slice selection only.
- Branch name and PR URL or planned PR slot:
  `nvidia-goal-status-rollup`;
  <https://github.com/uv-xiao/pto-cu/pull/142>.
- Allowed scope and files:
  `docs/in_progress/nvidia_backend/goal_status_rollup.md`,
  `docs/in_progress/nvidia_backend/dispatch_log.md`,
  `docs/in_progress/nvidia_backend/pr_slicing_plan.md`, and optionally a
  focused guard in `tests/ut/py/test_nvidia_review_artifacts.py`.
  The worker must not edit implementation code, `.agents/**`, CUDA examples,
  serving probes, or stable `docs/nvidia-backend/**`.
- Dependencies and blocked assumptions:
  worker starts from `origin/main` at
  `3ee1523cc37a6574b739d3e7c0a9060f55f2aea5`. It must cite current
  repo-relative evidence and PR/merge references, not conversation memory.
- Verification commands and results:
  initial monitor tick showed pane, transcript, and worktree all ok with
  `dirty_count: 0`. Worker verification passed: `git diff --check`,
  `git diff --cached --check`, targeted `markdownlint-cli2` over the three
  touched docs, NVIDIA review guard, and focused review-artifact tests with
  `59 passed`.
- Merge decision and merge commit:
  pending worker PR, parent review, and exact-head merge decision.
- Handoff summary and remaining gaps:
  worker produced the status rollup and selected
  `nvidia-moe-uccl-ep-fused-boundary-h200`: a reduced fused cross-GPU MoE
  boundary from the accepted persistent-MoE plus UCCL-EP handoff evidence.

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
