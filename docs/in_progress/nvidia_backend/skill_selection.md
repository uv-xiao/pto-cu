# Codex Skill Source Selection

This record converts the cloned skill sources from the restart source corpus
into a Codex-facing selection policy. The sources are references only; do not
bulk-copy them into `.agents/` without a focused adaptation pass and a
regression guard.

## Selection Summary

- `SihaoLiu/skills` `monitor-codex-goal`:
  adapted locally. The local `codex-goal-monitor` skill now supports
  Codex-to-Codex supervision through tmux, with a read-only tick helper for
  interval-based monitoring.
- `SihaoLiu/skills` `codex-pr-loop`:
  selected for later port. It matches a future GitHub PR babysitting loop, but
  the current repo first needs a small GitHub status cache or tick helper if
  polling becomes frequent.
- `SihaoLiu/skills` `superman`:
  selected as an orchestration reference. It captures large-feature flow, but
  local dispatch rules already define the Codex dispatcher/worker split.
- `PolyArch/humanize` `humanize-rlcr`:
  selected as a review-loop reference. The stop-hook RLCR shape is useful, but
  implementation should not import hook machinery without a focused need.
- `PolyArch/humanize` `ask-codex`:
  selected as a command-shape reference. It is useful for safe quoted
  `codex exec` calls. Any local version must record outputs in repo-appropriate
  scratch paths.
- `obra/Superpowers`:
  selected by capability, not copied. Core skills are already available in the
  current Codex environment. Import source changes only when a missing behavior
  is identified.

## Codex Adaptation Rules

- Install only one adapted skill at a time, with a test or command that proves
  the repository entry point exists and names its Codex-specific assumptions.
- Preserve the user-visible intent of the source skill, but replace
  source-platform-specific tools before installation. Examples include remote
  control tools, push notifications, cron helpers, plugin roots, stop hooks,
  and subagent assumptions that do not exist in Codex.
- Any Codex-to-Codex monitoring workflow must keep the target session read-only
  except for an explicit tmux injection path that requires human approval or a
  documented emergency criterion.
- Do not persist derived steering rules outside the monitor's own evidence
  files unless a human asks to promote them into `.agents/` or `docs/`.
- Keep `.agents/skills/` operational and short. Historical captures,
  transcripts, and downloaded source trees stay in `tmp/` or focused review
  docs, not in installed skill manuals.

## Next Installation Order

1. Add a GitHub PR status tick/cache helper only if parent monitoring keeps
   spending too many tokens on repeated `gh` or connector reads.
2. Adapt `codex-pr-loop` after the PR-status helper has a concrete local
   contract.
3. Distill more orchestration guidance only if the existing
   `ultimate-goal-dispatch` rule fails to cover a repeated dispatcher error.
