# 2026-06-01 VDCores RepeatM State Lite Diagnostic

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_repeat_state_lite_h200` to the paper-baseline
  execution-attempt data.
- Refreshed `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json` so the VDCores resource-policy blocker points at
  the latest diagnostic.
- Added focused review assertions for the decoded `RepeatM` fields and the
  bad PC50 accumulator evidence.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-state-lite-3cd4b6c7/
```

## Architecture Quality

The diagnostic was kept as a tmp-only VDCores patch and was not pushed to any
upstream repository. It narrowed the prior lane-source probe to lane-0-only
state printed through the existing debug macro, avoiding the broad per-lane
print path that perturbed the schedule.

The run keeps the PC48 writeback wait from the previous diagnostic so the
store-warp metadata hazard is controlled. It then shows that PC49 decodes as
the expected non-accumulating `RepeatM` for registers `0..4`, with size `2`
and delta `0x1000000000`. Immediately afterward, PC50 consumes source lane `0`
where lane-0 `gpr1` is `0`, but the native runtime `addr_accum` is
`0x7fffff`, which mutates desc33 coordinates to `(65535,127,0)`.

This moves the VDCores blocker from "maybe bad RepeatM encoding" to the
allocwarp RepeatM accumulator transport path. The prior split-shuffle variant
is still not a complete fix, so the next experiment should isolate native
64-bit shuffle/register transport and then test a narrow 32-bit
packed-coordinate transport.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

The launch command was:

```bash
CUDA_VISIBLE_DEVICES=7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  --debug-num-layers 1 \
  --debug-stop-after logits \
  -N 1 \
  --launch
```

The rebuild used the tmp-only patch in:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-state-lite-3cd4b6c7/vdcores-repeat-state-lite.patch
```

Key results:

- rebuild status: `0`;
- launch status: `1`;
- no `Unknown mem wb opcode` appeared after the PC48 wait;
- PC49 decoded `reg_start=0`, `reg_end=5`, `size=2`, `arg=0x0000`, and
  `address=0x1000000000`;
- after PC49, lane-0 `gpr0=0x1000000000` and `gpr1=0x0`;
- PC50 used source lane `0` and lane-0 `gpr1=0x0`, but `addr_accum=0x7fffff`;
- PC50 produced desc33 coordinates `(65535,127,0)`;
- final error: illegal instruction.

## Remaining Gaps

- Isolate native 64-bit shuffle/register transport in allocwarp with a
  non-perturbing reproduction.
- Test a lane-0-only split transport or explicit 32-bit packed-coordinate
  transport for RepeatM consumers before importing VDCores queue/resource
  policy timing.
