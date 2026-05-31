# 2026-05-31 CUTLASS Tensor-Tile Capture

## Code And Data Changed

- Added `cutlass_tensor_tile_capture.py` to capture a vendor CUTLASS
  `16x16x16` tensor-tile baseline through a normal `nvcc` build.
- Added the `cutlass` vendor-baseline method to the benchmark viewer.
- Imported A100 and H200 `tensor_core_tile` viewer rows from
  `tmp/cuda-backend/paper-baselines/cutlass/tensor-tile-4ab68198/`.
- Updated the tensor-core paper-evaluation matrix and regenerated
  `paper_readiness_audit.json`.

## Architecture Quality

The CUTLASS baseline is kept outside the PTO runtime implementation. The
capture script generates a small CUDA translation unit that instantiates
`cutlass::gemm::device::Gemm`, compiles it with `nvcc`, runs warmup and repeat
measurements, validates against a host reference, and converts raw JSON into
the same viewer result schema used by the PTO, direct CUDA, cuBLAS, Triton,
and paper-baseline rows.

## Evaluation Run

The fixture-based exporter test first failed because
`cutlass_tensor_tile_capture.py` did not exist. After implementation, A100 and
H200 captures each passed with 20 samples and `max_abs_error < 1.0e-3`; the H200
run used the remote tree-sync fallback plus a separate sync of
`tmp/baselines/cutlass/`.

## Remaining Gaps

- ThunderKittens still needs full upstream correctness and benchmark sweeps
  beyond the bounded MHA capture.
