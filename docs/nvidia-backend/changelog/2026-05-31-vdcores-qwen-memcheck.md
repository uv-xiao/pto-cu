# VDCores Qwen Memcheck Diagnostic

## Code And Data Changed

- Added the `vdcores_qwen3_1p7b_final_rms_memcheck_h200` execution attempt
  to the benchmark viewer data.
- Recorded the local raw artifact paths for the H200 `compute-sanitizer`
  memcheck run under
  `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-compute-sanitizer-f300a3b7/`.
- Extended the review-artifact test so the viewer must expose the sanitizer
  tool, exit status, error count, failing source locations, and blocker text.

## Architecture Quality

This separates the VDCores Qwen blocker from generic model-access or runtime
rebuild failures. The rebuilt runtime has the required Qwen compute exports and
the one-layer staged launch sweep shows `final_rms` is the earliest failing
stage. The memcheck run further localizes that first launch to the VDCores
load-warp path: `cp_async_bulk` reports invalid global reads through
`ldwarp_execute_singlethread`, then the warp reports an illegal address.

That evidence keeps the current paper-readiness status honest: VDCores is
imported, rebuilt, and launchable far enough to diagnose device code, but it is
not yet a valid correctness or resource-policy baseline for paper tables.

## Evaluation Run

- Hardware: H200 on the remote evaluation host.
- Baseline: local VDCores clone at commit `5247328`; push remote disabled.
- Command:

  ```bash
  compute-sanitizer --tool memcheck \
    python app/python/qwen3_1p7b/sched.py \
      --hf-cache-dir <shared-hf-cache> \
      --debug-num-layers 1 \
      --debug-stop-after final_rms \
      -N 1 --launch
  ```

- Result: sanitizer exit status `99`.
- Error summary: `130` errors.
- First invalid access: invalid `4096` byte global read in
  `cuda::ptx::cp_async_bulk` at `cp_async_bulk.h:62`, device frame
  `ldwarp_execute_singlethread`, VDCores source `ldwarp.cuh:52`, kernel source
  `dae2.cuh:167`, thread `(192,0,0)`.
- Follow-on device fault: warp illegal address in `ldwarp_execute_singlethread`
  at `ldwarp.cuh:113`.
- Host entry reporting the launch failure: `launch_dae`.

## Remaining Gaps

- Diagnose the VDCores tensor-map/load-warp address contract behind the
  `cp_async_bulk` invalid read.
- Re-run the one-layer Qwen correctness path after that fix or configuration
  correction.
- Capture the VDCores resource-policy trace and latency data only after the
  correctness path is no longer failing at first launch.
