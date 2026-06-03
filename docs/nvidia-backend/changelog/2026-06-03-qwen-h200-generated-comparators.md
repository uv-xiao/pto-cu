# 2026-06-03 Qwen H200 generated comparators

## Code And Data Changed

Added H200 three-repeat Triton and CUTLASS generated-kernel comparator
captures for the Qwen attention `16x64x128` tensor tile and Qwen MLP
`16x64x256` tensor tile. The generated-kernel coverage data now records both
A100 and H200 captures for each Qwen model-shape target:

- `tmp/cuda-backend/qwen-attention-generated-tensor-target-h200-repeat3-e1ab002b/`
- `tmp/cuda-backend/qwen-mlp-generated-tensor-target-h200-repeat3-e1ab002b/`

Fixed the CUTLASS tensor-tile capture script so `--arch compute_90` emits
`-gencode=arch=compute_90,code=compute_90` instead of pairing a Hopper
architecture with `code=compute_80`.

## Architecture Quality

The benchmark-viewer tensor workload guard now validates generated comparator
captures per hardware target. `a100_h200_multi_repeat` generated captures must
include both A100 and H200 entries, and each entry must provide Triton and
CUTLASS viewer records with matching hardware metadata, target tensor shape,
three samples, and correctness pass.

This leaves ThunderKittens as the explicit remaining comparator family for the
Qwen tensor targets instead of mixing missing hardware rows with missing method
rows.

## Evaluation Run

Synced the current tree to the remote H200 host, then ran:

```bash
.venv/bin/python .agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py \
  --device 0 --rows 16 --cols 64 --inner <128|256> --tile-count 1 \
  --warmup 3 --repeats 3 \
  --output tmp/cuda-backend/<target>/triton/capture.json \
  --artifact-root tmp/cuda-backend/<target>/triton/ \
  --viewer-output tmp/cuda-backend/<target>/triton/viewer-records.json
```

```bash
.venv/bin/python .agents/skills/cuda-backend-eval/scripts/cutlass_tensor_tile_capture.py \
  --device 0 --rows 16 --cols 64 --inner <128|256> --tile-count 1 \
  --warmup 3 --repeats 3 --arch compute_90 \
  --output tmp/cuda-backend/<target>/cutlass/capture.json \
  --artifact-root tmp/cuda-backend/<target>/cutlass/ \
  --viewer-output tmp/cuda-backend/<target>/cutlass/viewer-records.json
```

Observed viewer summaries:

- attention Triton: `sample_count=3`, correctness pass,
  `compute_target=compute_90`;
- attention CUTLASS: `sample_count=3`, correctness pass,
  `compute_target=compute_90`;
- MLP Triton: `sample_count=3`, correctness pass,
  `compute_target=compute_90`;
- MLP CUTLASS: `sample_count=3`, correctness pass,
  `compute_target=compute_90`.

RED checks first failed on the missing multi-hardware generated comparator
contract and the missing CUTLASS gencode helper. After implementation, the
focused guard passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::\
test_cutlass_tensor_capture_gencode_matches_requested_arch \
  tests/ut/py/test_nvidia_review_artifacts.py::\
test_tensor_workload_coverage_records_multi_repeat_qwen_capture \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

## Remaining Gaps

This narrows the tuned tensor workload gap but does not close it.
ThunderKittens comparator rows are still required for the Qwen tensor target
shapes before this gap can be promoted.
