# 2026-06-01 VDCores Qwen3 8B Global Instructions

## Code And Data Changed

- Added a VDCores Qwen3-8B execution attempt for the 64-token instruction
  capacity diagnostic.
- Added
  `docs/nvidia-backend/baseline-patches/vdcores-qwen3-8b-global-insts-16384.patch`
  as the reproducibility patch used for the temporary VDCores source variant.
- Updated the VDCores run contract, paper-readiness audit, work queue, serving
  workload blockers, and goal-progress data to point at the new correctness
  blocker instead of the older pre-launch capacity blocker.

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-instruction-capacity-0a0392d2/
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-global-insts-0a0392d2/
```

## Architecture Quality

The diagnostic separates capacity from correctness:

- the 64-token Qwen3-8B schedule needs up to `2177` compute instructions and
  `15042` memory instructions per SM, so the default `512`-instruction
  shared-load runtime cannot represent it;
- a temporary global-instruction runtime with `numInsts=16384` can build and
  run `-N 64 -b 5`;
- the same global-instruction runtime fails Qwen3-8B correctness thresholds,
  so it is not a valid paper baseline row.

The remote VDCores source was restored and the shared-instruction runtime was
rebuilt after the diagnostic.

## Evaluation Run

The temporary patch changed `include/dae/context.cuh` to disable shared
instruction loading and raise the global instruction table:

```text
dae2LoadInstructions = false
numInsts = 16384
```

The patched build completed on H200. `ptxas` reported `2336 bytes smem`,
compared with `14624 bytes smem` for the shared-instruction rebuild.

The paper-policy timing command completed:

```bash
DAE_BENCH_WARMUP=1 HF_TOKEN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  python app/python/qwen3/sched.py \
  --hf-cache-dir <shared-hf-cache>/hub -N 64 -b 5
```

It reported median execution time `303995744 ns` over five iterations.

Correctness under the same patched runtime failed `gate_proj_low`,
`up_proj_low`, `silu`, and `final_rms` thresholds, although the final token
still matched:

```text
[correctness] PASS final_token: ref=422, dae=422
RuntimeError: Correctness check failed
```

## Remaining Gaps

- Fix the VDCores global-instruction path correctness regression or implement
  an equivalent segmented/token-windowed Qwen3 schedule under the
  shared-instruction runtime.
- Re-run Qwen3-8B `-N 64 -b 5` with correctness passing before importing a
  VDCores paper-serving result row.
- Keep the global-instruction timing artifact marked as diagnostic evidence
  until it has matching correctness.
