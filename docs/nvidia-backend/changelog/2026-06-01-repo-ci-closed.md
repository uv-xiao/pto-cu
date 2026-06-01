# 2026-06-01 Repository CI Closed For Ultimate Goal

## Code And Data Changed

- Confirmed GitHub Actions are disabled at the `uv-xiao/pto-cu` repository
  settings level while the NVIDIA backend ultimate goal is active.
- Disabled the GitHub workflow entry named `NVIDIA Manual Review` for
  `uv-xiao/pto-cu`.
- Removed the runnable `.github/workflows/ci.yml` file from the branch so
  GitHub cannot register a repository workflow while the goal is active.
- Archived the future manual-review workflow recipe at
  `docs/ci/nvidia-manual-review.workflow.yml` for human reference.
- Updated the ultimate-goal work preparation policy so repository Actions must
  remain disabled during exploratory NVIDIA backend child slices.

## Architecture Quality

The standalone pto-cu project now treats local verification, benchmark
artifacts, changelog reports, and the dispatch log as the active progress gates
during the ultimate goal. The checked-in state contains no runnable workflow
YAML under `.github/workflows/`, which avoids inherited or stale repository CI
state blocking NVIDIA backend exploration before the CUDA review surface is
stable.

The synthetic GitHub `Dependency Graph` workflow still reports `active` through
the workflows API, but GitHub rejects disabling it with HTTP 422. Repository
Actions are disabled, so it is not an active PR gate.

## Evaluation Run

```bash
gh api repos/uv-xiao/pto-cu/actions/permissions
```

Result:

```json
{"enabled":false,"sha_pinning_required":false}
```

```bash
gh api repos/uv-xiao/pto-cu/actions/workflows
```

Result:

```text
NVIDIA Manual Review: disabled_manually
Dependency Graph: active
```

```bash
gh pr checks 1 --repo uv-xiao/pto-cu
```

Result:

```text
no checks reported on the 'main' branch
```

## Remaining Gaps

- Keep repository Actions disabled until the NVIDIA backend review and
  evaluation contract is stable enough to reintroduce non-blocking CI.
- If Actions are reopened, update this policy and add a new changelog report in
  the same review slice.
