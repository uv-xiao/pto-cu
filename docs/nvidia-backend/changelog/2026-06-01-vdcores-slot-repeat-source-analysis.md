# 2026-06-01 VDCores Slot/Repeat Source Analysis

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_slot_repeat_source_analysis_h200` to the
  paper-baseline execution-attempt data.
- Refreshed `paper_readiness_audit.json`, `paper_readiness_work_queue.json`,
  and `goal_progress.json` so the latest VDCores blocker points at the
  slot/repeat source analysis.
- Added focused review assertions for the slot metadata and invalid-coordinate
  evidence.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-slot-repeat-source-analysis-858717e4/
```

## Architecture Quality

The source analysis narrows the VDCores logits blocker from "inspect slot
allocation and `RepeatM`" to two concrete runtime hazards in the direct GEMV
window:

- `pc48` allocates the `desc34` `storeC` writeback into slot `0`.
- `pc49` emits the `RepeatM` that controls the next direct GEMV iteration.
- `pc50` reuses slot `0` for the `desc33` `loadB` and overwrites
  `st_insts[0]` before the store warp consumes the `pc48` writeback metadata.
- The store warp then reports `Unknown mem wb opcode: slot_mask=1 slot=0
  op=12 opcode=0301`, proving it read load metadata where store metadata was
  expected.
- The same path feeds `desc33` coordinates `(65535,127,0)`, while the
  build-only schedule expects `[0,0,0,0]` for that load.

This keeps the paper-readiness evidence honest: VDCores still has no valid
queue/resource-policy timing, but the next experiment now has a precise
runtime hypothesis instead of a broad logits-stage failure.

## Evaluation Run

This was a source/log diagnostic, not a new remote launch. It used the prior
co-located H200 logits split-6 failure log plus the build-only logits schedule
introspection artifact:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-logits-split-bisect-cec118fe/logits-split-6.log
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-logits-schedule-introspection-e5d6786f/logits-schedule-summary.json
```

The source evidence is under `tmp/baselines/vdcores/` and includes the
allocator, allocation warp, store warp, load warp, GEMV task, `RepeatM`, and
`SchedGemm` paths. No upstream repository was edited or pushed.

## Remaining Gaps

- VDCores still has no correctness result or queue/resource-policy timing for
  the persistent-device scheduler comparison.
- The next VDCores diagnostic should instrument allocation-warp GPR lanes and
  slot reuse around `pc48` through `pc51`, then test a VDCores-local change
  that prevents writeback-slot metadata reuse before the store warp has copied
  the `MInst`.
