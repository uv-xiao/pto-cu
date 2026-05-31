# 2026-05-31 Readiness Audit Run Readiness

## Code And Data Changed

- Extended `paper_readiness_audit.py` to read
  `paper_baseline_run_readiness.json`.
- Added `paper_baseline_run_readiness_statuses` to each audit claim, filtered
  to paper-baseline runs that are not yet `imported_to_viewer`.
- Rendered claim-level run-readiness statuses in the benchmark viewer.
- Updated the benchmark-viewer validator, review guard, focused tests, CUDA
  evaluation workflow, and shared contracts to require the new audit field.

## Architecture Quality

The paper-readiness audit now owns the final human-review blocker list instead
of requiring reviewers to cross-check the separate run-readiness table by hand.
Readiness rows still remain pre-run evidence only: non-pass run-readiness
records create blockers, but they do not count as measured benchmark results
or promote baseline runs.

## Evaluation Run

The audit integration was added with a failing fixture first:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_readiness_audit_matches_current_viewer_data'
```

The committed audit was regenerated with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
    --output docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json
```

Current claim blockers now include MPK/VDCores `HF_TOKEN` and VDCores
`dae.runtime` readiness gaps, plus vLLM and SGLang paired-probe dependency
gaps, directly inside `paper_readiness_audit.json`.

## Remaining Gaps

The audit is still `not_paper_ready`. The next paper-readiness movement needs
actual imported raw baseline results: measured MPK/VDCores scheduler runs,
serving captures for MPK, VDCores, vLLM, and SGLang, and broader
ThunderKittens correctness/benchmark sweeps.
