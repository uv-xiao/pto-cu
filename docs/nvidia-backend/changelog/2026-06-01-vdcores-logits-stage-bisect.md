# 2026-06-01 VDCores Logits Stage Bisect

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_logits_stage_bisect_h200` to the paper-baseline
  execution-attempt data.
- Regenerated paper-readiness audit, work-queue, and goal-progress data so the
  latest VDCores blocker points at the logits-stage failure.
- Updated review-artifact tests to require the new VDCores stage-bisect
  evidence.

Raw diagnostic sources are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-stage-bisect-cec118fe/
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-logits-split-bisect-cec118fe/
```

## Architecture Quality

The previous diagnostic showed that single-visible-GPU placement fixes the
earliest `final_rms` failure. This diagnostic narrows the remaining co-located
failure to the first stage after `final_rms`: adding `logits_proj` is enough to
make VDCores fail before any paper-grade resource-policy capture.

The default `QWEN1P7B_LOGITS_SPLIT_M=6` reaches the VDCores kernel and fails
with illegal instruction. The raw debug logs also show invalid slot allocation
or invalid TMA-coordinate signals in the logits-era work. Lower split values
`1`, `2`, and `3` do not prove a runtime workaround because VDCores rejects
those shapes before launch with its auto-folding placement assertion.

No upstream repository was edited or pushed.

## Evaluation Run

The stage bisect ran:

```bash
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after <logits|argmax|restore|full> \
  -N 1 \
  --launch
```

Result: all four stage cuts exited with status `1`. Because the prior
single-visible-GPU `final_rms` cut exited with status `0`, the first failing
co-located stage is `logits`.

The logits split sensitivity ran:

```bash
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=<1|2|3|6> \
HF_HOME=<shared-hf-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after logits \
  -N 1 \
  --launch
```

Result: split values `1`, `2`, and `3` failed before launch on the VDCores
auto-folding placement assertion; split value `6` reached the kernel and
failed with illegal instruction.

## Remaining Gaps

- VDCores still has no correctness or queue/resource-policy timing result for
  the persistent-device scheduler-overhead comparison.
- The next diagnostic must inspect logits projection scheduling, SM placement,
  and slot allocation before attempting paper-grade VDCores timing.

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
