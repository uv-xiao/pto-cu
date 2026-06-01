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

## Qwen Runtime Input Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_runtime_input_binding.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-input-binding/qwen-runtime-input-binding.json
```

Expected output: command exits 0; output JSON records padded target-length
Qwen `input_ids`, matching `attention_mask`, decode `output_ids` capacity,
prompt alignment status, and scalar bindings for the MPK and VDCores serving
policies.

This is a host-side runtime input artifact, not a CUDA allocation. It turns
the tokenizer output into `runtime_token_buffer_plan`,
`attention_mask_buffer`, and `decode_output_buffer_plan`; CUDA token-buffer
allocation and decode-loop consumption remain runtime gaps.

## Qwen CUDA Token Buffer Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_token_buffer_binding.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-token-buffer/qwen-cuda-token-buffer-binding.json
```

Expected output: command exits 0; output JSON records CUDA allocation and copy
verification for Qwen input_ids, attention_mask, and output_ids token buffers
when the host runtime is available.

The artifact allocates those buffers, copies host token data to device memory,
verifies copy-back, and leaves decode-loop consumption as the remaining
runtime gap.

## Qwen Persistent Decode Arguments

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_decode_args.py \
  --output-json tmp/cuda-backend/pto-serving-decode-args/qwen-persistent-decode-args.json
```

Expected output: command exits 0; output JSON records how Qwen token device
pointers bind to persistent DAG a/b/out fields while preserving tensor_args
for weights.

This is a persistent decode task-argument artifact. Without a live token
pointer table from the decode-loop runner it records symbolic pointer sources;
with a token pointer table it validates concrete `a`, `b`, and `out` bindings.

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

## Qwen Safetensors Shard Status

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_fetch.py \
  --output-json tmp/cuda-backend/pto-serving-shards/qwen-safetensors-shards.json
```

Expected output: command exits 0; output JSON records Qwen safetensors shard
URLs, local target paths, present/missing counts, and resumable fetch commands
without downloading by default.

This is a placement and fetch-plan artifact, not a CUDA loader. Use
`--download` only when intentionally fetching the 16 GB Qwen3-8B safetensors
shards into `tmp/sources/qwen3-8b-safetensors/`; rerun the metadata probe after
all shards report `present`.

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

## Qwen CUDA Weight Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --output-json tmp/cuda-backend/pto-serving-weight-binding/qwen-cuda-weight-binding.json
```

Expected output: command exits 0; output JSON records stable CUDA binding
slots, safetensors file byte ranges, persistent-device readonly weight
argument roles, and bounded or full CUDA residency probe status.

This is a binding artifact, not a full model loader. With local CUDA runtime
libraries available it copies a bounded subset of small tensors to device
memory through the existing runtime C API, then frees them. Full weight
residency can be probed explicitly:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_weight_binding.py \
  --cuda-probe-mode full \
  --device 4 \
  --verify-tensors 16 \
  --output-json tmp/cuda-backend/pto-serving-weight-residency/qwen-cuda-weight-residency.json
```

The full mode keeps all copied tensors resident until the whole model is
loaded, verifies selected small tensors by copying bytes back, then frees every
allocation. Persistent task-argument pointer binding remains a runtime gap.

## Qwen Persistent Weight Arguments

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_args.py \
  --output-json tmp/cuda-backend/pto-serving-weight-args/qwen-persistent-weight-args.json
```

Expected output: command exits 0; output JSON records Qwen weight task
descriptors whose `tensor_args` fit the four-pointer persistent DAG ABI and
cover every validated weight tensor.

This is an ABI manifest, not runtime pointer materialization. It decomposes
Qwen layer work into persistent task descriptors such as attention QKV,
attention Q/K norm, MLP gate/up, and MLP down so each task stays within
`PtoCudaPersistentDagTask::tensor_args[4]`.

## Qwen Persistent Weight Materialization

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_materialization.py \
  --output-json tmp/cuda-backend/pto-serving-weight-materialization/qwen-persistent-weight-materialization.json
```

Expected output: command exits 0; output JSON records how Qwen persistent
weight task descriptors are materialized through the `CudaPersistentDagTask`
ctypes layout, and binds resident device pointers when a live pointer table is
supplied.

Without `--pointer-table-json`, this emits a symbolic materialization plan that
uses `resident_weight_ptrs[slot_id]` as the source for each `tensor_args`
entry. With a live pointer table from the decode-loop runner, it emits concrete
device addresses and validates that each pointer matches the expected tensor
slot before DAG submission.

## Qwen Resident Weight Table

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_resident_weight_table.py \
  --output-json tmp/cuda-backend/pto-serving-resident-weight-table/qwen-resident-weight-table.json
```

Expected output: command exits 0; output JSON records the process-scoped
resident weight pointer owner lifecycle, materialization bridge, pointer
count, and teardown count; add `--cuda-live` to allocate and copy through the
CUDA runtime.

The default mode is `dry_run_pointer_lifecycle`: it exercises the same owner,
pointer-table, materialization, and close/free ordering without allocating
16.38 GB again. Use `--cuda-live` only inside a runner process that will submit
the persistent DAG while the owner is still open.
