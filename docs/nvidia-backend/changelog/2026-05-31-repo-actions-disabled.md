# 2026-05-31 Repository Actions Disabled

## Code And Data Changed

- Disabled GitHub Actions for `uv-xiao/pto-cu` at the repository settings
  level while the NVIDIA backend ultimate goal is active.
- Updated the CI policy to distinguish the checked-in manual workflow from the
  repository-level Actions switch.
- Kept `.github/workflows/ci.yml` as a manual review recipe for the point where
  Actions are deliberately reopened.

## Architecture Quality

The standalone pto-cu branch now has two CI guardrails:

- repository GitHub Actions are disabled, so checks cannot block goal progress;
- checked-in workflows remain manual-only, so reopening Actions cannot
  accidentally reintroduce inherited Ascend a2a3/a5 push or PR jobs.

This keeps progress gates local and reviewable: changelog reports, dispatch-log
entries, review guards, focused tests, and explicit benchmark artifacts.

## Evaluation Run

```bash
gh api repos/uv-xiao/pto-cu/actions/permissions --jq '{enabled, allowed_actions}'
```

Result:

```json
{"allowed_actions":null,"enabled":false}
```

## Remaining Gaps

- GitHub Actions are unavailable until a reviewer deliberately reopens them.
- Any future reopening must update `docs/ci.md`, the review guard if automatic
  triggers are allowed, and a changelog report in the same slice.
