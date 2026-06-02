# Paper Baseline Source Provenance Guard

## Code And Data Changed

- Tightened benchmark-viewer validation so every paper-baseline source path
  must be under `tmp/`, exist locally, be a git clone, match the committed
  pinned commit, and contain a README or INSTALL document.
- Updated the validator call path so paper-baseline validation can inspect the
  repo-local review workspace.

## Architecture Quality

MPK, VDCores, vLLM, SGLang, and ThunderKittens paper-baseline records now prove
that their local source checkouts are present and pinned, rather than only
recording intended source metadata.

## Evaluation Run

- `validate_benchmark_viewer_data.py` passed against the current
  `tmp/baselines/` source clones with the new provenance checks.
- Focused benchmark-viewer, changelog, review-ready, pytest, and diff checks
  passed after the guard was added.

## Remaining Gaps

The source clones are inspectable and pinned. Paper-grade rows still require
the remaining work-queue runs and imports before the LLM-serving claim can be
paper-ready.
