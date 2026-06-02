# Qwen Shape Field Materialization Guard

## Code And Data Changed

- Added a freshness check for default Qwen persistent weight-args artifacts in
  `qwen_persistent_weight_materialization_impl.loaders`.
- If the default artifact under `tmp/` lacks callable `task_shape_fields`, the
  materializer now regenerates weight args from
  `examples/cuda/qwen_persistent_weight_args.py` instead of silently consuming
  the stale file.
- Added a focused regression for stale shape-field artifacts.

## Architecture Quality

The resource-backed Qwen path now treats shape fields as part of the runtime
contract, not optional metadata. This prevents stale local artifacts from
routing logits back to the scalar diagnostic fallback after the source builder
has learned the Qwen3-8B callable shapes.

The guard keeps generated runtime descriptors aligned with the source-level
task-shape contract without committing bulky refreshed `tmp/` artifacts.

## Evaluation Run

Focused tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  -q -k 'weight_args_loader_rejects_shape_field_stale_artifact or materialized_weight_descriptor_preserves_task_shape_fields or qwen_weight_descriptors_emit_callable_shape_fields or launch_packet_carries_cuda_task_shape_fields'
```

Materialization evidence was written under
`tmp/cuda-backend/pto-serving-shape-field-materialization-72b2cfc9/`. The
logits descriptor reported:

```text
cols=151936, inner=4096, lda=4096, ldb=4096, ldc=151936, scalar0=256
```

## Remaining Gaps

A live full-vocab tiled-logits diagnostic run was attempted but not imported:
even a one-step resource-backed full-vocab scalar-loop projection took too
long for this iteration. PTO still needs an efficient tiled/tensor-core logits
kernel path before importing this as full-serving or paper-ready evidence.
