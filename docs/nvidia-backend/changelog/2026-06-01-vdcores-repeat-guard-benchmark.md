# 2026-06-01 VDCores RepeatM Guard Benchmark

## Code And Data Changed

- Added `vdcores_qwen3_1p7b_repeat_guard_bench_h200` to the
  paper-baseline execution-attempt data.
- Imported a reviewable H200 viewer result for
  `llm_serving_decode` / `vdcores` with median execution time
  `1778528 ns`, average execution time `1779008 ns`, and five benchmark
  iterations on 132 H200 SMs.
- Marked `vdcores_resource_policy_trace` as `captured_raw` and refreshed the
  paper-readiness audit, work queue, goal-progress data, and focused review
  tests.

Raw artifacts remain under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-native-f6b16bac/
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-full-offline-f6b16bac/
tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-bench-f6b16bac/
```

## Architecture Quality

The VDCores patch stayed tmp-only and was not pushed upstream. It guards the
allocwarp `addr_accum` shuffle so the shuffle runs only when `RepeatM` is
active and the source lane is valid. The diagnostic also keeps the narrow
writeback-slot wait used to control the PC48/PC55 metadata lifetime hazard.

The guarded trace supports this root cause: the unguarded runtime computes
`addr_accum` before `RepeatM` is active at PC49, using invalid source lane 48.
With the guard, PC49 skips the invalid shuffle, the first active PC50/PC51/PC52
uses source lane 0 and address `0x0`, and the next repeat uses address
`0x1000000000`. The full one-token offline launch and benchmark then complete.

This narrows the VDCores paper-readiness gap from "debug the logits-stage
crash" to a smaller baseline-integration task: prove correctness on the
guarded path and add queue-pressure plus scheduler-overhead metadata comparable
with PTO persistent-device and MPK.

## Evaluation Run

Remote host: `bizhaoh200`, H200, `CUDA_VISIBLE_DEVICES=7`.

The full launch used offline Hugging Face cache flags:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  -N 1 \
  --launch
```

The benchmark used the same guarded VDCores runtime and:

```bash
DAE_BENCH_WARMUP=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
QWEN1P7B_NO_PREFETCH=all \
QWEN1P7B_LOGITS_SPLIT_M=6 \
python app/python/qwen3_1p7b/sched.py \
  --hf-cache-dir <shared-hf-cache> \
  -N 1 \
  -b 5
```

Key results:

- patch apply status: `0`;
- rebuild status: `0`;
- full launch status: `0`;
- benchmark status: `0`;
- patch restore status: `0`;
- minimum execution time: `1774240 ns`;
- median execution time: `1778528 ns`;
- average execution time: `1779008 ns`;
- maximum execution time: `1785504 ns`.

The first full-launch attempt without offline Hugging Face flags stalled in
HEAD retries and was terminated with status `143`; the offline rerun completed.
Both local and remote VDCores checkouts were restored clean after capture.

## Remaining Gaps

- Run guarded VDCores single-token correctness and import the result.
- Capture queue-pressure and scheduler-overhead metadata comparable with PTO
  persistent-device and MPK.
- Repeat latency distributions on the final patched baseline policy before
  using the VDCores row as paper-ready evidence.
