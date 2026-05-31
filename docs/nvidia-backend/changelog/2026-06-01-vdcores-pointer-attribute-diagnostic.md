# 2026-06-01 VDCores Pointer Attribute Diagnostic

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_pointer_attr_probe_h200` to the
  benchmark-viewer execution-attempt data.
- Regenerated paper-readiness audit, work-queue, and goal-progress data.
- Updated review-artifact tests so the latest VDCores action points at the
  pointer-attribute diagnostic instead of the previous load-warp-only run.

Raw diagnostic sources are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-pointer-attrs-sm64-c6e25d3e/
```

## Architecture Quality

The probe uses a temporary wrapper under `tmp/` that monkey-patches
VDCores' Python `dae_app` entrypoint for one process. It does not edit or push
the VDCores upstream repository. The wrapper lets the existing Qwen3-1.7B
schedule build normally, then inspects the built memory instructions before
launch.

This adds a new fact to the VDCores blocker: the sampled direct 1D effective
addresses are readable by ordinary `cudaMemcpy` before launch, but
`cuMemGetAddressRange` cannot classify them. That makes the remaining failure
more specific: VDCores still fails in `cp_async_bulk`, but the sampled
prelaunch direct-load addresses are not simply unreadable CUDA pointers.

No upstream repository was edited or pushed.

## Evaluation Run

The H200 probe used the public Qwen3-1.7B path with the same one-layer
`final_rms` schedule:

```bash
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VDC_POINTER_ATTR_OUTPUT=<artifact-root>/pointer-attrs-summary.json \
VDC_PROBE_SMS=64,65,66,67,68,69,70,71,0,1,2,3,4,5,6,7 \
VDC_PROBE_MAX_DIRECT=160 \
python <artifact-root>/vdcores_pointer_attr_probe.py
```

Result: probe status `0`. It sampled 80 direct 1D load/store effective
addresses. `cuMemGetAddressRange` returned status `201` for all 80, while
device-to-host `cudaMemcpy` succeeded for all 80. The SM64 pc1
`OP_ALLOC_TMA_LOAD_1D` effective address copied 4096 bytes successfully before
launch after applying the token-52 runtime accumulator.

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
- The next diagnostic should combine same-process pointer classification with
  the failing launch, or inspect why `cp_async_bulk` and compute-sanitizer
  treat `cudaMemcpy`-readable direct 1D addresses as out of bounds.
