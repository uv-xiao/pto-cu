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

## Persistent Qwen Serving Scaffold

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
  --output-json tmp/cuda-backend/pto-serving-scaffold/qwen-serving-scaffold.json
```

Expected output: command exits 0; output JSON records `status=partial` until
Qwen tokenizer, weight loader, KV-cache lifecycle, decode-loop runner, and
full-serving viewer import stages exist.

This is not a benchmark result. It is the repo-owned lifecycle scaffold for
the PTO `Qwen/Qwen3-8B` full-serving work queue item.

## Qwen Serving Lifecycle Plan

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_serving_lifecycle_plan.py \
  --output-json tmp/cuda-backend/pto-serving-lifecycle/qwen-serving-lifecycle-plan.json
```

Expected output: command exits 0; output JSON records the Qwen3-8B model
shape, KV-cache capacity ladder, weight-binding plan, and persistent-device
task mapping for the MPK and VDCores serving policies.

This is a lifecycle contract artifact, not a full-serving result. It makes the
memory and callable mapping reviewable before tokenizer, safetensors loading,
kernel bodies, decode-loop execution, and viewer-result import are complete.
