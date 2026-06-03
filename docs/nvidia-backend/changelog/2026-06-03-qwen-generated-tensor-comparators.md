# 2026-06-03 Qwen generated tensor comparators

## Code And Data Changed

Added A100 three-repeat generated-kernel comparator evidence for the Qwen
attention `16x64x128` tensor tile and Qwen MLP `16x64x256` tensor tile.
`evaluations/nvidia/benchmark-viewer/data/tensor_workload_coverage.json` now
records `generated_kernel_capture` entries for Triton and CUTLASS, with raw
artifact roots and viewer record paths under:

- `tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/`
- `tmp/cuda-backend/qwen-mlp-generated-tensor-target-a100-repeat3-c743cb84/`

The Triton and CUTLASS tensor-tile capture scripts now label viewer records
as `n=1024, tensor tile <rows>x<cols>x<inner>`, so generated comparator rows
can be validated against the Qwen model-shape tile instead of the old generic
diagnostic tensor shape.

## Architecture Quality

The benchmark-viewer guard now validates generated-kernel comparator captures
as a separate contract from PTO/cuBLAS Graph throughput captures. The guard
requires Triton and CUTLASS records, A100 `compute_80` hardware metadata,
three samples, correctness pass, concrete tensor-tile shape labels, and
artifact paths under the declared raw roots.

The shared command validator still requires `--single-baseline` markers for
PTO/cuBLAS Graph commands, while allowing generated-kernel scripts to use
their direct `--viewer-output` export path.

## Evaluation Run

Captured A100 comparator rows with:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py \
  --device 0 --rows 16 --cols 64 --inner 128 --tile-count 1 \
  --warmup 3 --repeats 3 \
  --output tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/triton/capture.json \
  --artifact-root tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/triton/ \
  --viewer-output tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/triton/viewer-records.json
```

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cutlass_tensor_tile_capture.py \
  --device 0 --rows 16 --cols 64 --inner 128 --tile-count 1 \
  --warmup 3 --repeats 3 --arch compute_80 \
  --output tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/cutlass/capture.json \
  --artifact-root tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/cutlass/ \
  --viewer-output tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/cutlass/viewer-records.json
```

The same commands were run with `--inner 256` and the `qwen-mlp` artifact
root for the MLP tile.

RED tests first failed on missing shape-label helpers and missing
`generated_kernel_capture` evidence. After implementation, the focused guard
run passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_generated_tensor_capture_scripts_label_model_shape_tiles \
  tests/ut/py/test_nvidia_review_artifacts.py::test_tensor_workload_coverage_records_multi_repeat_qwen_capture \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

Observed capture summaries:

- attention Triton: `sample_count=3`, correctness pass,
  `device_wall_ns=63487`;
- attention CUTLASS: `sample_count=3`, correctness pass,
  `device_wall_ns=8192`;
- MLP Triton: `sample_count=3`, correctness pass, `device_wall_ns=77983`;
- MLP CUTLASS: `sample_count=3`, correctness pass, `device_wall_ns=9216`.

## Remaining Gaps

This narrows the tuned tensor workload gap but does not close it. H200
multi-repeat rows and ThunderKittens comparator rows are still required for
the Qwen tensor tiles before the paper-ready tensor workload evidence can be
promoted.
