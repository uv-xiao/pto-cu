# 2026-06-01 VDCores Device Colocation Diagnostic

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_launch_pointer_attr_probe_h200` to record
  same-process pointer attributes immediately before the failing launch.
- Added `vdcores_qwen3_1p7b_single_visible_gpu_h200` to record the
  single-visible-GPU follow-up.
- Regenerated paper-readiness audit, work-queue, and goal-progress data.
- Updated review-artifact tests so the latest VDCores blocker is the
  co-located one-layer full-schedule failure.

Raw diagnostic sources are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-launch-pointer-attrs-ac8dab26/
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-single-visible-gpu-ac8dab26/
```

## Architecture Quality

The same-process probe shows the original VDCores `final_rms` launch was not
just using opaque pointers. CUDA Runtime recognized all 80 sampled direct
addresses with `cudaPointerGetAttributes`, and host `cudaMemcpy` could read
them before launch. The important difference is placement: sampled direct
loads included device-7 pointers while the VDCores kernel launched on current
device 0; sampled direct stores were on device 0.

The follow-up constrains H200 visibility with `CUDA_VISIBLE_DEVICES=7`, so the
Qwen weights and VDCores launcher buffers are co-located on logical device 0.
That makes the previously failing one-layer `final_rms` launch return status
0. The next blocker is later in the broader schedule: the one-layer full
schedule still fails with illegal instruction.

No upstream repository was edited or pushed.

## Evaluation Run

The same-process pointer/launch probe used the one-layer `final_rms` schedule:

```bash
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VDC_LAUNCH_POINTER_ATTR_OUTPUT=<artifact-root>/launch-pointer-attrs-summary.json \
python <artifact-root>/vdcores_launch_pointer_attr_probe.py
```

Result: 80 sampled direct addresses were recognized by
`cudaPointerGetAttributes`, all 80 copied with device-to-host `cudaMemcpy`,
direct-load pointers spanned devices 0 and 7, and the launch raised
`launch_dae failed: an illegal memory access was encountered`.

The single-visible-GPU follow-up ran:

```bash
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after final_rms \
  -N 1 \
  --launch
```

Result: status `0`.

The same placement was then used for the one-layer full schedule:

```bash
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after full \
  -N 1 \
  --launch
```

Result: status `1`, with `launch_dae failed: an illegal instruction was
encountered`.

## Remaining Gaps

- VDCores still has no correctness or queue/resource-policy timing result for
  the persistent-device scheduler-overhead comparison.
- The next diagnostic must debug the co-located one-layer full-schedule
  illegal instruction before running full VDCores correctness or timing.

## Verification

Passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
jq empty docs/nvidia-backend/benchmark-viewer/data/*.json
git diff --check
```
