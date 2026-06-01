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

## Qwen Prompt Accounting

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_prompt_accounting.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-tokenizer/qwen-prompt-accounting.json
```

Expected output: command exits 0; output JSON records tokenizer class,
chat-template status, observed prompt-token counts, target deltas, and whether
padding or prompt regeneration is required for the MPK and VDCores serving
policies.

Offline mode requires the Qwen tokenizer to be available in the local cache.

Use `--mode download` only when intentionally capturing tokenizer evidence
from Hugging Face into `tmp/`. Use `--mode mock` for dependency-free local
contract checks; mock output must not be imported as paper evidence.

## Qwen Weight Inventory

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_weight_inventory.py \
  --output-json tmp/cuda-backend/pto-serving-weights/qwen-weight-inventory.json
```

Expected output: command exits 0; output JSON records safetensors shard count,
tensor count, binding groups, total size, the config-derived expected
shape/dtype contract, and the remaining tensor-open and CUDA weight-binding
gaps.

This is a weight inventory, not a weight loader. It parses the Qwen3-8B
safetensors index captured under `tmp/sources/` and makes the persistent-device
binding groups and expected tensor shapes reviewable before any model tensors
are opened or copied.

## Qwen Safetensors Metadata Probe

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_metadata.py \
  --weight-inventory-json tmp/cuda-backend/pto-serving-weights/qwen-weight-inventory.json \
  --output-json tmp/cuda-backend/pto-serving-safetensors/qwen-safetensors-metadata.json
```

Expected output: command exits 0; output JSON records whether Qwen
safetensors shard headers were opened and whether actual tensor shapes/dtypes
match the expected contract.

This is a metadata probe, not a CUDA loader. When the Qwen shards are absent it
reports `shards_missing`; when shards are present it parses standard
safetensors headers and validates tensor shape/dtype metadata before any data
copy or CUDA binding step.
