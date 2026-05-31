# 2026-05-31 VDCores MInst Provenance Diagnostic

## Code And Data Changed

- Added the `vdcores_qwen3_1p7b_minst_provenance_h200` execution attempt to
  the benchmark viewer data.
- Regenerated paper-readiness audit, work-queue, and goal-progress artifacts
  from the viewer data.
- Added review-artifact test coverage for the new VDCores diagnostic fields:
  sampled SMs, token row provenance, direct-load ownership, and next blocker.

Raw diagnostic sources are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-minst-provenance-83e8135b/
```

## Architecture Quality

The VDCores blocker is now narrower. The previous no-prefetch sweep only
showed that per-stage async prefetch routing was not the sole cause of the
`final_rms` failure. This diagnostic maps generated `MInst` direct 1D loads
back to live Python-side tensor ranges before launch.

The viewer now records that SM64-SM71 generate the embedding row load through
`OP_CC0` plus a jumped `OP_ALLOC_TMA_LOAD_1D`, and that SM0-SM7 generate the
RMS and hidden direct 1D loads from known tensor ranges. That shifts the next
root-cause search to runtime/device-side mutation, instruction upload, or
CUDA memcheck allocator visibility for direct 1D `cp_async_bulk` loads.

No upstream repository was edited.

## Evaluation Run

The H200 diagnostic captured a plain instruction dump for SM68:

```bash
QWEN1P7B_NO_PREFETCH=all \
  python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after final_rms \
  -N 1 -i 68
```

It also captured a structured `runpy` provenance dump without launch, using
the same one-layer `final_rms` schedule and sampled SMs 0-7 and 64-71.

The focused test was run before the viewer record existed and failed because
`vdcores_qwen3_1p7b_minst_provenance_h200` was absent from
`paper_baseline_execution_attempts.json`. After adding the record and
regenerating derived artifacts, the focused test passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  -q
```

Result: `1 passed`.

## Remaining Gaps

- VDCores still has no paper-grade correctness or resource-policy result for
  Qwen3 1.7B.
- The next diagnostic must inspect runtime/device-side mutation, instruction
  upload, or CUDA memcheck allocator visibility for direct 1D `cp_async_bulk`
  loads.
