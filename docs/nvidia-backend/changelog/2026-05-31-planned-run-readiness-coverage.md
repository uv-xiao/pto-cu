# 2026-05-31 Planned Run Readiness Coverage

## Code And Data Changed

- Generalized `paper_baseline_run_readiness.py` from the two scheduler-trace
  runs to every paper-baseline run that is not already `imported_to_viewer`.
- Added paired-probe status as an explicit readiness check, so vLLM and SGLang
  dependency blockers now appear in the benchmark viewer next to their planned
  serving runs.
- Regenerated `paper_baseline_run_readiness.json` and updated validators,
  review guards, and focused tests to require this broader run coverage.

## Architecture Quality

Run readiness is now a single review surface for all pending paper-baseline
work, not a scheduler-only side path. This keeps pre-run blockers visible for
MPK, VDCores, vLLM, and SGLang while preserving the rule that readiness is not
benchmark evidence and cannot promote a run to `imported_to_viewer`.

## Evaluation Run

The broader behavior was added with a failing fixture first:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_run_readiness_probe_exports_run_blockers'
```

The current viewer readiness data was regenerated with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py \
    --output-root tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-1ace72fb \
    --viewer-output docs/nvidia-backend/benchmark-viewer/data/paper_baseline_run_readiness.json
```

Current vLLM readiness is `partial` because the paired probe reports missing
`vllm` module availability on A100 and H200. Current SGLang readiness is
`partial` because the paired probe reports unresolved SGLang module/import
paths for the selected benchmark entrypoints.

## Remaining Gaps

These records still do not replace measured serving or scheduler baseline
results. MPK, VDCores, vLLM, and SGLang runs must still produce raw artifacts
under `tmp/cuda-backend/paper-baselines/`, flow through
`paper_baseline_results_update.py`, and update the paper-readiness audit.
