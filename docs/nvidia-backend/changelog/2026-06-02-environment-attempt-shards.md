# Environment Attempt Shards

## Code And Data Changed

- Replaced the committed `paper_baseline_environment_attempts.json` payload
  with `data/paper_baseline_environment_attempts/index.json` and one record
  file per bounded environment setup attempt.
- Extended the shared viewer-data sharding helper for
  `paper_baseline_environment_attempts`.
- Updated the environment-attempt generator so real viewer writes use the
  sharded collection path while explicit temporary test outputs remain flat.
- Updated the HTML viewer and focused review test to load environment attempts
  through the sharded manifest.

## Architecture Quality

The setup-attempt evidence is now reviewable per baseline and retry window.
This keeps long command-log metadata out of one large viewer JSON file while
preserving the logical collection contract used by validators and the viewer.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a data-layout cleanup only. It does not resolve the remaining MPK,
VDCores, or serving-capture paper-readiness gaps.
