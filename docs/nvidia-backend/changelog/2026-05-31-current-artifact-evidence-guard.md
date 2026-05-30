# 2026-05-31 Current Artifact Evidence Guard

## Code And Data Changed

Tightened `.agents/checks/validate_benchmark_viewer_data.py` so current
benchmark evidence cannot be represented by a path string alone. The validator
now requires current result artifacts, current paper-matrix `raw_artifact`
evidence, and latest paired probe roots to resolve under `tmp/` and contain
JSON evidence.

Updated the focused review tests to assert the same artifact existence rule
against the committed viewer data, and updated the shared contract to
distinguish current evidence from planned `expected_artifacts`.

## Architecture Quality

This strengthens the code-document evidence boundary. Planned baseline outputs
can still be documented as future `expected_artifacts`, but any path used to
support a current viewer result or paper-matrix status must be inspectable in
the local review workspace.

The rule keeps the benchmark viewer useful for human review: a reviewer can
follow a current raw-artifact path and find JSON evidence instead of only a
declared destination.

## Evaluation Run

The existing current artifacts satisfy the stricter guard:

- `tmp/cuda-backend/current-head-full-layered-cross-fixed/combined-current-743709f3/`
- `tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/`
- `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`
- `tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-67c5c655/`
- `tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/`

The validation command is:

```bash
.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

This guard proves the current evidence paths are inspectable, but it does not
turn planned MPK, VDCores, vLLM, SGLang, or full ThunderKittens sweeps into
paper results. Those systems still need raw captures under `tmp/` before their
rows can move from planned artifacts to current evidence.
