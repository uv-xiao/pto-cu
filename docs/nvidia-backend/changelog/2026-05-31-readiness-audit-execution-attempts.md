# Readiness Audit Execution Attempts

## Code And Data Changed

- Extended `paper_readiness_audit.py` so blocked paper-baseline claims include
  the latest execution attempt for each non-imported run.
- Extended `paper_readiness_work_queue.py` so diagnostic work items preserve
  `execution_attempt_id`.
- Updated `refresh_nvidia_review_artifacts.py`,
  `paper_baseline_results_update.py`, and the benchmark-viewer validator to
  treat `paper_baseline_execution_attempts.json` as an audit input.
- Regenerated `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json`.

## Architecture Quality

The paper-readiness queue no longer loses the most useful diagnostic evidence.
After the MPK and VDCores sanitizer runs, the persistent-device scheduler
claim now shows both generic missing paper-grade rows and concrete diagnostic
blockers:

- MPK: `mpk_qwen3_0p6b_token1_memcheck_h200`, a scheduler-side null write
  through `paged_kv_indices_snapshot`.
- VDCores: `vdcores_qwen3_1p7b_final_rms_memcheck_h200`, invalid
  `cp_async_bulk` reads through the load-warp path.

This keeps the human-reviewable work queue aligned with the latest evidence
instead of asking reviewers only to "run the baseline" after the baseline has
already failed with actionable source locations.

## Evaluation Run

The focused red/green test passed after regenerating the derived artifacts:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_readiness_audit_matches_current_viewer_data or paper_readiness_work_queue_matches_current_audit'
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

## Remaining Gaps

- MPK and VDCores diagnostic attempts still fail after kernel launch, so the
  persistent-device scheduler claim remains blocked.
- The next implementation work is to resolve those baseline-specific
  diagnostic blockers or document why the paper comparison must use a
  different workload shape.
