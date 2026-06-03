# 2026-06-03 Qwen H200 tensor throughput

## Code And Data Changed

Added H200 three-repeat PTO persistent-device and cuBLAS Graph throughput
captures for the Qwen attention `16x64x128` tensor tile and Qwen MLP
`16x64x256` tensor tile. The structured coverage data now records
multi-hardware throughput captures for both model-shape targets:

- `tmp/cuda-backend/qwen-attention-tensor-target-h200-repeat3-85187ffd/`
- `tmp/cuda-backend/qwen-mlp-tensor-target-h200-repeat3-85187ffd/`

The tensor workload coverage validator now accepts a `captures` list for
multi-hardware throughput evidence while preserving the old single-capture
shape for existing records.

## Architecture Quality

The new guard requires `a100_h200_multi_repeat` throughput captures to include
both A100 and H200 hardware entries. Each entry must have a current artifact
root, exported viewer records under that root, matching hardware metadata,
the target Qwen tensor shape, correctness pass, and the shared sample count.

This keeps backend evidence and paper-result evidence separated: H200 PTO and
cuBLAS Graph rows are now present, while generated CUTLASS/Triton H200 rows
and ThunderKittens comparator rows remain explicit open work.

## Evaluation Run

Synced the current tree to the remote H200 host, then ran samples with
`CUDA_HOME=/usr/local/cuda-12.8`, `compute_90`, and
`PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda`.

For each Qwen tensor tile, three samples were captured for:

```bash
.venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
  --device 0 --sizes 1024 --repeats 1 --arch compute_90 \
  --single-baseline pto_persistent_dag_graph_tensor_core \
  --tensor-rows 16 --tensor-cols 64 --tensor-inner <128|256>
```

```bash
.venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
  --device 0 --sizes 1024 --repeats 1 --arch compute_90 \
  --single-baseline cublas_sgemm_graph \
  --tensor-rows 16 --tensor-cols 64 --tensor-inner <128|256>
```

The raw H200 captures were exported with `cuda_viewer_export.py` into
`viewer-records.json` for each tile.

Observed viewer summaries:

- attention PTO persistent-device: `sample_count=3`, correctness pass,
  median `device_wall_ns=56960`;
- attention cuBLAS Graph: `sample_count=3`, correctness pass,
  median `device_wall_ns=10271`;
- MLP PTO persistent-device: `sample_count=3`, correctness pass,
  median `device_wall_ns=68576`;
- MLP cuBLAS Graph: `sample_count=3`, correctness pass,
  median `device_wall_ns=12032`.

The RED test first failed because the throughput contract still expected only
`a100_multi_repeat`. After implementation, the focused guard passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::\
test_tensor_workload_coverage_records_multi_repeat_qwen_capture \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

## Remaining Gaps

This narrows the tuned tensor workload gap but does not close it. H200
generated CUTLASS/Triton rows and ThunderKittens comparator rows are still
required for the Qwen tensor tiles.
