# 2026-06-01 VDCores Queue/Scheduler Trace

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_queue_scheduler_h200` to the paper-baseline
  execution-attempt data.
- Imported a VDCores `paper_baseline_scheduler_trace` viewer row for
  `vdcores_resource_policy_trace`.
- Marked `vdcores_resource_policy_trace` as `imported_to_viewer`.
- Replaced the earlier missing queue/scheduler artifact blocker with a narrower
  paper-readiness gap: repeat the trace on the final non-diagnostic baseline
  policy before paper use.
- Refreshed the paper-readiness audit, work queue, and goal-progress data.

Raw artifacts are preserved under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-queue-scheduler-46872fa4/
```

## Architecture Quality

The VDCores source changes stayed tmp-only and were restored from both local
and H200 baseline checkouts after capture. The committed repo only records
distilled review data and the raw artifact path.

The imported scheduler metric is the resident allocwarp scheduler lifetime from
VDCores profile slots. It overlaps worker execution and must not be read as
additive latency. The queue-pressure metric is shared-memory slot allocator
pressure, which is the VDCores resource bottleneck behind the memory/compute
queue flow.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

The successful benchmark command shape was:

```bash
DAE_COMPUTE_OPS_FILE=<artifact>/qwen3-1p7b-compute-ops.vdcore.build \
EXTRA_NVCC_FLAGS="-include cfloat -I<cutlass-include> \
  -DDAE_DIAG_GUARD_REPEAT_SHUFFLE \
  -DDAE_DIAG_WAIT_AFTER_WB_ALLOC" \
make PYTHON=<project-venv-python> pyext

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
DAE_BENCH_WARMUP=1 \
<project-venv-python> app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  -b 5
```

Key results:

- benchmark status: `0`;
- correctness status from a separate fresh process: `0`;
- benchmark iterations: `5`;
- median execution time: `1787488 ns`;
- average execution time: `1786304 ns`;
- mean allocwarp scheduler resident time per SM: `1763558 ns`;
- max live slot occupancy: `24 / 24`;
- mean live slot pressure: `0.9958`;
- mean allocation retry cycles per SM: `198923 ns`;
- correctness checks: `17`;
- final token: `ref=25, dae=25`.

## Remaining Gaps

- Repeat the VDCores queue/scheduler trace on the final non-diagnostic baseline
  policy before treating this row as paper-ready evidence.
- Scale from the Qwen3-1.7B bring-up path to the Qwen3-8B paper target when
  VDCores baseline support is ready.
