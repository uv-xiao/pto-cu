# 2026-06-01 VDCores RepeatM Lane-Source Diagnostic

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_repeat_lane_source_diagnostic_h200` to the
  paper-baseline execution-attempt data.
- Refreshed `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json` so the VDCores resource-policy blocker points at
  the latest diagnostic.
- Added focused review assertions for the split-shuffle negative result and
  the refreshed paper-readiness audit action.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-lane-source-diagnostic-53fd525a/
```

## Architecture Quality

The diagnostic was kept as a tmp-only VDCores patch and was not pushed to any
upstream repository. It tested one hypothesis from the prior slot-lifetime
run: whether the bad `RepeatM` accumulator was caused by native 64-bit shuffle
lowering.

The result rules out that simple fix. Forcing a split 32-bit shuffle avoids
the earlier `pc50` desc33 coordinate mutation, but the same run shows native
and split shuffle values disagreeing across lanes for the same source lane.
The forced split variant then corrupts a later desc32 address and fails with
illegal memory access.

This keeps the VDCores paper-readiness state honest: queue/resource-policy
timing is still blocked, and the next debug target is the `RepeatM` active-mask
and producer-lane contract rather than a direct replacement of the native
64-bit shuffle.

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
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-lane-source-diagnostic-53fd525a/vdcores-repeat-lane-source-diagnostic.patch
```

Key results:

- rebuild status: `0`;
- launch status: `1`;
- no `Unknown mem wb opcode` appeared after the PC48 wait;
- with the split-shuffle variant, PC50 updated address `0x0` and desc33
  coordinates `(0,0,0)`;
- PC52 then updated address `0x20000331000000` and produced desc32 coordinates
  `(0,12544,3)`;
- final error: illegal memory access.

## Remaining Gaps

- Instrument `RepeatM` active masks, source-lane ownership, producer-lane
  validity, and packed coordinate-delta lifetime without changing
  `addr_accum` semantics.
- VDCores still needs a correctness-passing Qwen path before queue/resource
  policy timing can be imported into the paper-ready comparison.
