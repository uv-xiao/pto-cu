# 2026-05-31 MPK Diagnostic Variants

## Code And Data Changed

- Added three MPK persistent-device diagnostic attempts to the benchmark viewer
  execution-attempt data:
  `mpk_qwen3_0p6b_launch_blocking_token2_h200`,
  `mpk_qwen3_0p6b_no_cutlass_token2_h200`, and
  `mpk_qwen3_0p6b_token1_noprofile_h200`.
- Added a local raw diagnostic summary under
  `tmp/cuda-backend/paper-baselines/mpk/debug-463686af/` so reviewers can
  inspect the H200 run statuses without parsing every log manually.
- Extended the NVIDIA review-artifact test to assert the new MPK attempt IDs,
  status classifications, blocker text, and key summary fields.

## Architecture Quality

The viewer now distinguishes three different MPK failure surfaces instead of
collapsing them into one generic persistent-kernel failure:

- launch-blocking mode reaches MPK graph construction and build artifacts, but
  stalls before a launch/synchronize result is logged;
- disabling the MPK CUTLASS kernel path still launches and fails with CUDA
  illegal memory access;
- reducing generation to one token still launches and fails with CUDA illegal
  memory access.

This keeps the paper-readiness claim honest: the added evidence strengthens
the blocker diagnosis, but it does not promote MPK persistent execution to a
paper-grade imported result.

## Evaluation Run

On the H200 host, the bounded MPK diagnostic sweep ran Qwen/Qwen3-0.6B through
three persistent variants:

```bash
CUDA_LAUNCH_BLOCKING=1 python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B --max-new-tokens 2 \
  --max-seq-length 128 --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 --ignore-eos --use-mirage

python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B --max-new-tokens 2 \
  --max-seq-length 128 --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 --ignore-eos --use-mirage \
  --no-use-cutlass-kernel

python demo/qwen3/demo.py \
  --model Qwen/Qwen3-0.6B --max-new-tokens 1 \
  --max-seq-length 128 --max-num-batched-requests 1 \
  --max-num-batched-tokens 1 --ignore-eos --use-mirage
```

Each variant generated the same 7261-task, 1870-event MPK graph. The
launch-blocking variant was terminated after stalling at MPK initialization.
The no-CUTLASS and one-token variants compiled and launched the persistent
kernel asynchronously, then failed at `torch.cuda.synchronize()` with CUDA
illegal memory access.

## Remaining Gaps

- MPK still lacks a paper-grade persistent-kernel run imported into the viewer.
- The persistent scheduler-overhead paper claim still carries the MPK missing
  evidence gap until a matching successful or otherwise paper-usable run is
  captured.
