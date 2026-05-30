# CUDA Examples

These examples are thin wrappers around the CUDA smoke paths used by the
NVIDIA backend evaluation. They are intentionally close to the benchmark
commands so reviewers can connect examples, docs, and artifacts directly.

## Host-Schedule Vector Ops

- Benchmark id: `host_schedule_vector_ops`
- Runtime: `cuda/host_schedule`
- Method id: `pto_host_schedule`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/host_schedule_vector_ops.py \
  --op add --n 1024 --arch compute_80
```

Expected output: command exits 0; optional output JSON records `status=pass`
for the selected host-schedule vector operation.

Use `--op` to select the evaluated host-schedule ABI shape:
`add`, `mul`, `scale`, `square`, `axpy`, `affine`, `triad`, `quad`,
`generic_args`, or `generic_args4`.

## Persistent Layered-Cross Graph

- Benchmark id: `graph_layered_cross`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_layered_cross.py \
  --n 1024 --arch compute_80 --scheduler-blocks 3
```

Expected output: command exits 0; optional output JSON records `status=pass`
and the `graph_descriptor_layered_cross` DAG shape.

This runs the same `graph_descriptor_layered_cross` smoke shape that feeds the
current `743709f3` benchmark gate.
