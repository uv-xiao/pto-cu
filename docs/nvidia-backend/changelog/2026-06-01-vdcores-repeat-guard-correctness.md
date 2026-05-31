# 2026-06-01 VDCores RepeatM Guard Correctness

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_repeat_guard_correctness_h200` to the
  paper-baseline execution-attempt data.
- Updated the VDCores H200 viewer result to mark correctness as `pass` and
  point at the guarded correctness artifact.
- Narrowed the persistent-device scheduler-overhead work item from
  "correctness plus queue-pressure and scheduler-overhead" to queue-pressure
  plus scheduler-overhead metadata.
- Refreshed the paper-readiness audit and work queue, then updated focused
  review tests for the new latest VDCores execution attempt.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-correctness-712f88e8/
```

## Architecture Quality

The VDCores source patch stayed tmp-only and was not pushed upstream. The run
used the same guarded RepeatM runtime policy as the benchmark slice: skip the
allocwarp `addr_accum` shuffle until `RepeatM` is active and the source lane
is valid, while keeping the narrow writeback-slot wait for the PC48/PC55
metadata lifetime hazard.

The H200 rebuild used the project venv, the selected Qwen3-1.7B compute-op
list, pinned CUTLASS headers, and the existing `-include cfloat` workaround.
Earlier environment attempts in the same artifact root show why those knobs
are required: non-interactive `python` was missing, system `pip` was blocked
by PEP 668, and `nvcc` needed both CUTLASS headers and `FLT_MAX`.

The passing result means guarded RepeatM is no longer only a launch/timing
workaround. It is now proven against full-layer Qwen3-1.7B single-token
correctness. The remaining persistent-device paper blocker is narrower:
VDCores still needs queue-pressure and scheduler-overhead metadata comparable
with PTO persistent-device and MPK.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

The successful command shape was:

```bash
EXTRA_NVCC_FLAGS="-include cfloat -I<cutlass-include> \
  -DDAE_DIAG_GUARD_REPEAT_SHUFFLE \
  -DDAE_DIAG_WAIT_AFTER_WB_ALLOC" \
make PYTHON=<project-venv-python> pyext

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
<project-venv-python> app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --correctness
```

Key results:

- patch apply status: `0`;
- rebuild status: `0`;
- correctness status: `0`;
- patch restore status: `0`;
- correctness checks: `17`;
- final token: `ref=25, dae=25`;
- all logged checks passed, including logits slices and final token agreement.

Both local and remote VDCores checkouts were clean after capture.

## Remaining Gaps

- Capture VDCores queue-pressure metadata comparable with PTO persistent-device
  and MPK.
- Capture VDCores scheduler-overhead metadata comparable with PTO
  persistent-device and MPK.
- Repeat latency distributions on the final patched VDCores baseline policy
  before treating the row as paper-ready evidence.
