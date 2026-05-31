# 2026-05-31 VDCores No-Prefetch Sweep

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_no_prefetch_sweep_h200` to
  `paper_baseline_execution_attempts.json`.
- Regenerated `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json` so the latest VDCores blocker points to this
  diagnostic.
- Extended the focused review-artifact test to require the no-prefetch sweep,
  the all-stage memcheck summary, and the generated work-queue action.

## Architecture Quality

The VDCores paper-baseline path now separates an async-prefetch hypothesis from
the remaining scheduler/load-address problem. The failed variants prove that
disabling named Qwen3-1.7B prefetch stages does not make the earliest
`final_rms` cut paper-grade.

The next reviewable blocker is narrower: inspect generated `MInst` load
addresses, coordinates, and tensor descriptors for the earliest `final_rms`
schedule before importing VDCores resource-policy or latency rows.

## Evaluation Run

The H200 run swept:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  QWEN1P7B_NO_PREFETCH=<variant> \
  python app/python/qwen3_1p7b/sched.py \
    --hf-cache-dir <shared-hf-cache> \
    --debug-num-layers 1 --debug-stop-after final_rms -N 1 --launch
```

It covered `all`, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_low`,
`gate_high`, `up_low`, `up_high`, and `down_proj`. Every variant loaded the
model and failed after `launch_dae`.

The all-stage no-prefetch memcheck also ran:

```bash
QWEN1P7B_NO_PREFETCH=all compute-sanitizer --tool memcheck \
  python app/python/qwen3_1p7b/sched.py \
    --hf-cache-dir <shared-hf-cache> \
    --debug-num-layers 1 --debug-stop-after final_rms -N 1 --launch
```

It still reported invalid 4096-byte `cp_async_bulk` global reads with an error
summary count of 18. The prior final-rms memcheck had 130 errors, so the
no-prefetch path changes the failure surface but does not clear it.

The focused TDD check first failed because the execution-attempt data did not
contain `vdcores_qwen3_1p7b_no_prefetch_sweep_h200`. After adding the record
and refreshing generated data, it passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  -q
```

## Remaining Gaps

- VDCores still has no imported paper-grade resource-policy, correctness, or
  latency row.
- The next diagnostic must inspect generated `MInst` load-address and tensor
  descriptor provenance for the earliest `final_rms` launch.
- The raw no-prefetch logs remain local `tmp/` review evidence and are not
  committed.
