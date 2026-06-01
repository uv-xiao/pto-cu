# 2026-06-01 VDCores Qwen3 8B Rebuild And Correctness

## Code And Data Changed

- Added a second VDCores Qwen3-8B H200 execution attempt that records the
  runtime rebuild, correctness preflight, token-1 timing run, and 64-token
  serving-policy blocker.
- Updated `vdcores_qwen3_8b_decode_preflight` expected artifacts to include
  the rebuilt runtime logs and the normalized summary.
- Refreshed run-readiness, paper-readiness audit, work queue, and goal
  progress data so the queue now points at the 64-token instruction-capacity
  blocker rather than the previous missing-operator blocker.

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-runtime-rebuild-4778a9a1/
```

## Architecture Quality

The VDCores evidence now separates three states that matter for review:

- Qwen3-8B operator coverage can be built by selecting the launcher-reported
  `DAE_COMPUTE_OPS` set.
- The rebuilt runtime passes the bounded Qwen3-8B correctness path.
- The full 64-token VDCores paper-serving policy is still blocked before
  launch by instruction-buffer capacity.

That keeps the paper matrix conservative while removing the stale
missing-operator blocker.

## Evaluation Run

The successful rebuild used the captured Qwen3-8B operator list, the existing
CUTLASS include path, and the `-include cfloat` compile workaround:

```bash
cd tmp/baselines/vdcores
CPATH=<cutlass>/include \
DAE_COMPUTE_OPS_FILE=<artifact>/qwen3-8b-compute-ops.vdcore.build \
make clean pyext \
  NVCC_FLAGS="-O3 -Iinclude/dae -Iinclude -Ibuild/generated \
  -I<cutlass>/include -std=c++20 -Xptxas=-v -use_fast_math \
  -lineinfo -DNDEBUG -include cfloat"
```

The bounded correctness run passed and logged:

```text
[correctness] PASS final_token: ref=422, dae=422
[correctness] all checks passed
```

The token-1 timing run emitted five-iteration benchmark statistics with a
median execution time of `4536640 ns`.

The paper-policy `-N 64 -b 5` command still fails before launch:

```text
assert len(self.cinsts) <= ctensor.shape[0]
```

## Remaining Gaps

- Increase or restructure the Qwen3 schedule instruction capacity so the
  64-token paper decode policy can build instructions.
- Re-run the VDCores Qwen3-8B `-N 64 -b 5` path and import latency,
  throughput, correctness, batch policy, and raw output artifacts.
- Keep the token-1 benchmark out of final paper claims; it is bounded
  runtime evidence, not the VDCores paper-serving row.
