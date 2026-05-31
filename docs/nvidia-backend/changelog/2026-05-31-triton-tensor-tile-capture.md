# 2026-05-31 Triton Tensor-Tile Capture

## Code And Data Changed

- Added `triton_tensor_tile_capture.py` to capture a generated-kernel
  `16x16x16` tensor-tile baseline with Triton `tl.dot`.
- Added the `triton` generated-kernel method to the benchmark viewer.
- Imported A100 and H200 `tensor_core_tile` viewer rows from
  `tmp/cuda-backend/paper-baselines/triton/tensor-tile-0054494c/`.
- Updated the tensor-core paper-evaluation matrix and regenerated
  `paper_readiness_audit.json`.

## Architecture Quality

The Triton baseline is kept outside the PTO runtime implementation. It writes
raw samples under `tmp/`, then emits the same viewer result schema as the PTO,
direct-launch, cuBLAS, and paper-baseline rows. The capture records hardware,
Triton/Torch versions, per-repeat host and CUDA-event timing, and correctness
against `torch.matmul` so the document claim has explicit raw evidence.

## Evaluation Run

The fixture-based exporter test first failed because
`triton_tensor_tile_capture.py` did not exist. After implementation, A100 and
H200 captures each produced 20 samples with `max_abs_error < 1.0e-3`; the
focused exporter pytest target passed.

## Remaining Gaps

- CUTLASS or CuTe has not been captured for the same tile shape.
- ThunderKittens still needs full upstream correctness and benchmark sweeps
  beyond the bounded MHA capture.
