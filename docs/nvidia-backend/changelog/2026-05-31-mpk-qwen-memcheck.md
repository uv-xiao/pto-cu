# MPK Qwen Memcheck Diagnostic

## Code And Data Changed

- Added the `mpk_qwen3_0p6b_token1_memcheck_h200` execution attempt to the
  benchmark viewer data.
- Recorded local raw artifacts for the H200 `compute-sanitizer` memcheck run
  under
  `tmp/cuda-backend/paper-baselines/mpk/compute-sanitizer-712240f8-offline/token1_noprofile/`.
- Extended the review-artifact test so the viewer must expose the sanitizer
  tool, offline-cache setting, generated task counts, first invalid access,
  and host/device source locations.

## Architecture Quality

This keeps the MPK persistent-kernel baseline status precise. Earlier MPK
attempts proved that the native torch path runs, MPK compiles, the generated
megakernel launches, and the smallest one-token persistent path still fails at
`torch.cuda.synchronize()`. The memcheck run localizes that failure to the
scheduler path rather than to CUTLASS kernels or profiling export.

The first invalid device access is a 4-byte global write through address
`0x0` in `prepare_next_batch` at `persistent_kernel.cuh:273`, reached from
`execute_scheduler` at `persistent_kernel.cuh:1235` inside
`scheduler_kernel`. The write targets `paged_kv_indices_snapshot`, so the next
MPK diagnosis should verify how the persistent-kernel runtime initializes that
snapshot pointer for offline Qwen runs.

This is diagnostic evidence only. It does not promote MPK to a paper-grade
persistent scheduler baseline because correctness, scheduler overhead, and
resource-policy metrics still do not complete.

## Evaluation Run

- Hardware: H200 on the remote evaluation host.
- Baseline: local Mirage MPK clone at commit `bde2dec`; push remote disabled.
- Command:

  ```bash
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  compute-sanitizer --tool memcheck --target-processes all \
    python demo/qwen3/demo.py \
      --model Qwen/Qwen3-0.6B \
      --max-new-tokens 1 \
      --max-seq-length 128 \
      --max-num-batched-requests 1 \
      --max-num-batched-tokens 1 \
      --ignore-eos \
      --use-mirage \
      --output-dir ../../cuda-backend/paper-baselines/mpk/compute-sanitizer-712240f8-offline/token1_noprofile/build \
      --save-tokens ../../cuda-backend/paper-baselines/mpk/compute-sanitizer-712240f8-offline/token1_noprofile/tokens.json
  ```

- Result: sanitizer exit status `1`.
- Error summary: `4` errors.
- Generated graph: `7261` tasks and `1870` events.
- Init warning: `cudaDeviceSetLimit` returned `cudaErrorInvalidValue` from
  `init_persistent_kernel`.
- First invalid access: invalid 4-byte global write in `prepare_next_batch` at
  `persistent_kernel.cuh:273`, thread `(0,0,0)`, block `(11,0,0)`, address
  `0x0`.
- Device stack: `execute_scheduler` at `persistent_kernel.cuh:1235`, then
  `scheduler_kernel` at `persistent_kernel.cuh:1391`.
- Host launch entry: `launch_persistent_kernel`.
- Host sync failure: `torch.cuda.synchronize`.

## Remaining Gaps

- Verify the MPK offline runtime allocation and initialization contract for
  `paged_kv_indices_snapshot`.
- Re-run the one-token persistent path after the scheduler-side null write is
  addressed or configured away.
- Capture MPK correctness, scheduler overhead, dispatch trace, resource policy,
  and raw latency artifacts before importing MPK as a paper table row.
