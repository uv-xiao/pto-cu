# 2026-06-01 Qwen Runtime Input Binding

## Code And Data Changed

- Added `examples/cuda/qwen_runtime_input_binding.py`, which converts the Qwen
  tokenizer output into host-materialized `input_ids` buffers, decode
  `output_ids` capacity, prompt-alignment metadata, and scalar bindings for
  the MPK and VDCores serving policies.
- Wired the runtime input-binding artifact into the Qwen serving scaffold,
  PTO serving preflight, CUDA example manifest, example README,
  benchmark-viewer matrix, in-progress paper-readiness docs, dispatch log, and
  review-artifact tests.
- Stabilized `refresh_nvidia_review_artifacts.py` so committed viewer refresh
  JSON uses `current-working` for review-only readiness artifacts instead of a
  transient pre-commit SHA.
- Captured current input-binding evidence at
  `tmp/cuda-backend/pto-serving-input-binding-2026-06-01/qwen-runtime-input-binding.json`.

## Architecture Quality

The runtime input binding separates tokenizer evidence from runtime buffer
ownership. Prompt accounting still records tokenizer availability and prompt
counts, while the new artifact records the concrete token IDs and buffer shapes
that a persistent decode-loop runner must copy to CUDA memory.

The artifact deliberately stops at `host_materialized_not_cuda_allocated`.
This makes the remaining boundary explicit: the decode-loop runner still has
to allocate CUDA token buffers, copy the host `input_ids`, and pass those
buffers through persistent task arguments.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_runtime_input_binding.py \
  --output-json \
  tmp/cuda-backend/pto-serving-input-binding-2026-06-01/qwen-runtime-input-binding.json
```

Result: `status=runtime_input_binding_plan_ready` with offline
`Qwen2TokenizerFast`; both target serving policies have `input_ids_buffer`,
`output_ids_buffer`, prompt-token checksums, batch-size ladders, and
decode-token capacities. The observed chat prompt has 18 tokens, so the MPK
64-token and VDCores 128-token paper policies still require target prompt
shape alignment before full-serving rows can be imported.

The focused TDD selector first failed because the runtime input-binding script
did not exist; after implementation the selected runtime input-binding,
preflight, and scaffold tests passed.

## Remaining Gaps

- Resolve target prompt shape alignment for the paper workload rows.
- Allocate CUDA token buffers, copy `input_ids`, and pass those buffers to the
  persistent decode loop.
- Bind real CUDA KV-cache buffers, run the resident weight table in
  `cuda_live` mode through DAG submission, generate Qwen kernels, execute the
  decode loop, and import full-serving viewer rows for `Qwen/Qwen3-8B`.
