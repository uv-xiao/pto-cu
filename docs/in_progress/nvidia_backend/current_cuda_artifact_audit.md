# Current CUDA Artifact Audit

This note records the current restart-artifact posture after the repository was
returned to normal `main`-based PR flow. It replaces the older dirty-orphan
workspace audit as the active slicing guide.

## Current State

- `main` contains the accepted CUDA/Gluon/serving evidence slices that landed
  through GitHub PRs.
- The root checkout should stay clean except for the one dispatcher branch
  currently being prepared for review.
- Historical orphan-workspace material is only a source for selective recovery.
  It is not a review branch and must not be recommitted wholesale.
- `tmp/` contains external source checkouts, raw command output, and scratch
  evidence. Those paths remain gitignored and are not committed.

## Preserve As Review Seeds

- Dispatcher tracking:
  `docs/in_progress/001-nvidia-backend.md`,
  `docs/in_progress/nvidia_backend/dispatch_log.md`, and slicing/source notes.
  These keep the long-running goal auditable without private scrollback.
- Remote CUDA runner:
  `.agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh`. This is the
  current reusable entry point for remote H200 commands.
- Codex monitor skill:
  `.agents/skills/codex-goal-monitor/`. This provides read-only tmux and
  transcript monitoring plus interval-based tick summaries for child sessions.
- Gluon generated-kernel path:
  `simpler_setup/gluon_gen.py`, CUDA Gluon examples, focused tests, and H200
  evidence docs. These represent accepted generator and correctness slices.
- Serving evidence probes:
  vLLM/DeepSeek probe examples and
  `docs/in_progress/nvidia_backend/vllm_*` notes. These record the serving path
  already explored through accepted PR slices.
- Review guards:
  `tests/ut/py/test_nvidia_review_artifacts.py` and
  `.agents/checks/check_nvidia_review_ready.py`. These keep review-facing docs,
  evidence claims, and selected agent surfaces synchronized.

## Reduce Or Retire Before Promotion

- Long terminal transcripts, tmux captures, raw benchmark dumps, and one-off
  generated output should stay under `tmp/`.
- Historical dispatcher logs should be summarized before commit. Do not
  restore a scrollback-sized dispatch log when a concise PR/merge record is
  enough.
- `.agents/` should keep reusable rules and scripts only. Capture-specific
  helpers belong in `tmp/` until their contract is clear.

## Not Decided In This Slice

- Whether any old orphan-workspace implementation files still contain useful
  seeds that are absent from current `main`.
- The next distributed communication implementation slice after the accepted
  Gluon and serving evidence work.
- The exact serving promotion path from in-progress evidence into
  `docs/nvidia-backend/`.

## PR Slicing Decision

Near-term work should continue in this order:

1. Land this restart tracking restoration branch so the parent session has a
   durable, reviewable operating record.
2. Audit the accepted `main` history against the umbrella goal and identify
   the next missing requirement with a PR-sized worker slice.
3. Launch one worker branch at a time unless the dispatch log records an
   explicit isolation strategy.
4. Keep monitor ticks interval-based and summary-first to avoid token-heavy
   tmux/GitHub polling.
