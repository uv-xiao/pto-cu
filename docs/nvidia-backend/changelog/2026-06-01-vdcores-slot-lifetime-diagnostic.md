# 2026-06-01 VDCores Slot Lifetime Diagnostic

## Code And Data Changed

- Added the H200 execution attempt
  `vdcores_qwen3_1p7b_slot_lifetime_pc44_pc52_h200` to the benchmark
  viewer data.
- Refreshed the paper-readiness audit, work queue, and goal-progress data so
  the VDCores resource-policy blocker points at the latest diagnostic.
- Added focused review assertions for the new VDCores attempt.

## Architecture Quality

The diagnostic was kept as a tmp-only VDCores patch and was not pushed to any
upstream repository. It narrowed the test to SM64 PC44-PC52 after two broader
variants perturbed earlier startup paths.

The valid run separates two hazards in the direct logits GEMV window:

- protecting the PC48 writeback metadata lets STU copy the correct opcode
  `0443` before PC50 reuses slot 0;
- the launch still fails because PC49/PC50 `RepeatM` mutates the desc33 TMA
  coordinates to `(65535,127,0,0)`.

That means queue/resource-policy timing is still blocked, but the next debug
target is now the `RepeatM` lane-source and packed-coordinate encoding path
rather than writeback metadata lifetime alone.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

Raw artifact root:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-slot-lifetime-diagnostic-be367ca4/
```

The valid launch command was:

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
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-slot-lifetime-diagnostic-be367ca4/vdcores-slot-lifetime-diagnostic-pc44-pc52.patch
```

Key results:

- rebuild status: `0`;
- launch status: `1`;
- PC48 wait changed flags from `0x00fffffe` to `0x00ffffff`;
- STU copied slot 0 opcode `0443`;
- no `Unknown mem wb opcode` appeared after the wait;
- PC50 applied `addr_accum=0x7fffff` and produced desc33 coordinates
  `(65535,127,0,0)`;
- final error: illegal instruction.

## Remaining Gaps

- Instrument the PC49/PC50 `RepeatM` lane source to determine why lane 0 reads
  `addr_accum=0x7fffff` instead of the packed delta represented by PC49.
- Queue/resource-policy timing for VDCores remains unimportable until the
  logits-stage launch reaches correctness.
