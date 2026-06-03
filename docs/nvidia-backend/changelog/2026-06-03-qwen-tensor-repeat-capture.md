# 2026-06-03 Qwen Tensor Repeat Capture

## Code And Data Changed

- Fixed `.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py`
  so its `CudaPersistentDagTask` ctypes layout matches the current
  persistent-device ABI: five tensor pointers, five tensor dtype tags, four
  scalar slots, then tensor/scalar counts.
- Added a regression test that compares the smoke-runner DAG task layout
  against `simpler_setup.cuda_callable_compiler.CudaPersistentDagTask`.
- Added A100 three-repeat throughput-capture metadata for the Qwen
  `16x64x128` attention tile and `16x64x256` MLP tile in
  `tensor_workload_coverage.json`.
- Extended the benchmark-viewer validator so optional `throughput_capture`
  records must point at existing `tmp/` artifacts, include PTO rows, carry at
  least three samples, and match exported viewer-record hardware, shape,
  correctness, raw-artifact, and sample-count fields.

## Architecture Quality

The CUDA persistent-device ABI now has one review guard across the compiler
helper and the smoke runner. This prevents stale Python launch-packet layouts
from silently corrupting device-visible task metadata after ABI extensions.

The Qwen tensor workload coverage now separates three states:

- source/runtime routing for Qwen tensor function ids `7240` and `7241`;
- local A100 multi-repeat PTO and cuBLAS Graph evidence;
- remaining H200 and generated-kernel comparator work.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_cuda_persistent_smoke_dag_task_matches_compiler_abi \
  -q
```

Result: failed because the smoke runner still exposed `tensor_args[4]`.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_cuda_persistent_smoke_dag_task_matches_compiler_abi \
  tests/ut/py/test_nvidia_review_artifacts.py::test_tensor_workload_coverage_records_multi_repeat_qwen_capture \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

Result: `3 passed`.

The previously failing target sample now passes:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
  --device 0 --sizes 1024 --repeats 1 --arch compute_80 \
  --single-baseline pto_persistent_dag_graph_tensor_core \
  --tensor-rows 16 --tensor-cols 64 --tensor-inner 128
```

Result: `status=pass`, zero scheduler errors, and dispatch function ids
`[7240, 1, 2, 1]`.

The new raw captures are:

- `tmp/cuda-backend/qwen-attention-tensor-target-a100-repeat3-4b281f79/`
- `tmp/cuda-backend/qwen-mlp-tensor-target-a100-repeat3-4b281f79/`

Each exported viewer-record file contains PTO persistent-device and cuBLAS
Graph A100 rows with `sample_count=3` and correctness `pass`.

## Remaining Gaps

The tuned tensor workload gap remains open. H200 repeat captures and
CUTLASS/Triton/ThunderKittens comparator rows are still required before the
gap can move out of `status/remaining-gaps/`.
