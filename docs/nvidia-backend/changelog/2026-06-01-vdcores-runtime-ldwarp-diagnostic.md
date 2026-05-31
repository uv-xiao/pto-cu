# 2026-06-01 VDCores Runtime LD Warp Diagnostic

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_runtime_ldwarp_sm64_h200` to the
  benchmark-viewer execution-attempt data.
- Added a raw summary artifact for the H200 SM64 load-warp diagnostic under
  `tmp/cuda-backend/paper-baselines/vdcores/`.
- Regenerated paper-readiness audit, work-queue, and goal-progress data.
- Updated review-artifact tests so the latest VDCores action points at the
  runtime load-warp address blocker.

Raw diagnostic sources are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-runtime-ldwarp-sm64-16944235/
```

## Architecture Quality

This slice keeps the VDCores baseline read-only and uses only a temporary
debug-only header sync in the ignored `tmp/baselines/vdcores` checkout. The
header was restored after the H200 capture, so future baseline runs do not
inherit the instrumentation.

The diagnostic narrows the VDCores blocker again. The prior provenance run
showed generated Python-side `MInst` addresses were live before launch. The
new H200 run shows the runtime load warp consumes the same direct 1D address
that compute-sanitizer reports as the first invalid 4096-byte `cp_async_bulk`
read. The next root-cause target is therefore why that runtime-consumed
global address is outside sanitizer allocation tracking or otherwise unmapped
during launch.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 extension was rebuilt with `DAE_DEBUG_PRINT=64` and the selected
Qwen3-1.7B compute-operator contract:

```bash
CPATH=<repo>/tmp/baselines/cutlass/include \
CUDA_HOME=/usr/local/cuda-12.8 \
DAE_COMPUTE_OPS_FILE=<repo>/tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-contract/compute_ops_full.txt \
make -C tmp/baselines/vdcores clean pyext debug=64 \
  NVCC="nvcc -include cfloat"
```

The failing one-layer `final_rms` cut was run under compute-sanitizer:

```bash
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
compute-sanitizer --tool memcheck --target-processes all \
  python app/python/qwen3_1p7b/sched.py \
    --hf-cache-dir <shared-hf-cache> \
    --debug-num-layers 1 \
    --debug-stop-after final_rms \
    -N 1 \
    --launch
```

Result: build status `0`, sanitizer status `1`, and `ERROR SUMMARY: 18
errors`. The SM64 debug print reported `addr=0x761ba2034000` for the first
4096-byte direct 1D load; compute-sanitizer reported the same address as the
first invalid global read in `ldwarp.cuh:56`.

Focused verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py \
  -q
```

Result: `32 passed`.

The benchmark-viewer data, changelog, review-ready guard, and JSON validators
also passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
jq empty docs/nvidia-backend/benchmark-viewer/data/*.json
```

## Remaining Gaps

- VDCores still has no correctness or queue/resource-policy timing result for
  the persistent-device scheduler-overhead comparison.
- The next diagnostic must explain why the runtime-consumed direct 1D address
  is outside sanitizer allocation tracking or otherwise unmapped during
  launch.
