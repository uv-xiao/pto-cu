# 2026-06-03 ThunderKittens GEMM Compatibility Probe

## Code And Data Changed

Added
`.agents/skills/cuda-backend-eval/scripts/thunderkittens_gemm_compatibility_probe.py`
to inspect the checked-out ThunderKittens GEMM sources without modifying the
upstream checkout. The probe records BF16 H100 GEMM tile constraints, INT8 H100
GEMM comparability limits, and per-Qwen-target compatibility decisions.

`evaluations/nvidia/benchmark-viewer/data/tensor_workload_coverage.json` now
records `thunderkittens_gemm_compatibility_probe` under both Qwen tensor
targets, pointing at:

```text
tmp/cuda-backend/paper-baselines/thunderkittens/qwen-gemm-compatibility-88da0949/compatibility.json
```

The ThunderKittens method metadata now describes the current evidence as
attention-family proxy rows plus source-inspected GEMM compatibility evidence,
with same-GEMM-tile Qwen comparator rows still open.

## Architecture Quality

The benchmark-viewer validator now rejects Qwen tensor targets whose
ThunderKittens GEMM compatibility probe omits the current artifact, target tile
mapping, BF16 and INT8 entrypoint rows, or the explicit gap-preserving status.

This prevents proxy attention evidence from being mistaken for same-GEMM-tile
Qwen comparator evidence.

## Evaluation Run

RED tests first failed on the missing probe module and missing
`thunderkittens_gemm_compatibility_probe` fields. After implementation, the
focused guard run passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_thunderkittens_gemm_compatibility_probe_marks_qwen_tiles_incompatible \
  tests/ut/py/test_nvidia_review_artifacts.py::test_tensor_workload_coverage_records_multi_repeat_qwen_capture \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

Observed source-probe result:

- BF16 H100 GEMM: 64x64 base tile, default 128x256 output block, not exact for
  Qwen 16x64x128 or 16x64x256 tiles.
- INT8 H100 GEMM: requires INT8 and `Mb=128`, so it is not comparable to the
  current float/tensor-core Qwen tensor claim.

## Remaining Gaps

The ThunderKittens same-GEMM-tile comparator gap remains open. The next step is
to capture source-compatible ThunderKittens BF16 GEMM rows as scaled comparator
evidence, add a reviewed local wrapper experiment for exact Qwen tiles outside
the upstream checkout, or record a policy exception before closing the gap.
