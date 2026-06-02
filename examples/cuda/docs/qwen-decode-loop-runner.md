# CUDA Examples: Qwen Decode Loop Runner

## Qwen Decode Loop Runner

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock \
  --run-unit-math-live \
  --run-submission-smoke \
  --single-context-live-session \
  --run-resource-backed-smoke \
  --resource-backed-repeat-runs 3 \
  --token-cuda-live \
  --kv-cuda-live \
  --resident-cuda-live \
  --workspace-cuda-live \
  --device 0 \
  --arch compute_80 \
  --repeat-runs 3 \
  --output-json tmp/cuda-backend/pto-serving-decode-loop-submission-descriptors/qwen-decode-loop-runner.json
```

Expected output: command exits 0; output JSON records a single CUDA-context
resource session, runner-owned cuda_live token, KV-cache, resident-weight, and
activation-workspace owners, resource-backed Qwen submission descriptors,
compact graph materialization, workspace-bound launch-packet preflight,
diagnostic bridge contracts, a diagnostic Qwen descriptor smoke execution, and
repeated diagnostic resource-backed `run_prepared` execution.

The artifact composes token pointer, KV-cache, and resident-weight owners into
a decode-loop submission plan. It records owner open/materialize/submit/close
ordering plus output-token accounting, and maps the owner-owned `a`, `b`,
`out`, `c`, `d`, and `tensor_args` fields into the repeated proxy live runner.
With `--run-unit-math-live`, it also executes the repeated unit-math
diagnostic from the runner entry point and records the bridge summary. With
`--token-cuda-live`, it opens the process-scoped token pointer-table owner in
the runner. With `--kv-cuda-live`, it opens the full planned KV-cache owner
in the runner. With `--resident-cuda-live`, it opens the resident weight-table
owner and materializes 399 weight pointers for the submission plan. With
`--workspace-cuda-live`, it allocates per-workload float32 activation buffers,
a logits/sampling output buffer, and runtime-generated RoPE cos/sin tables.
The launch-packet preflight resolves `runtime_buffers.rope_cos_table` and
`runtime_buffers.rope_sin_table` descriptor references into
`CudaPersistentDagTask::tensor_args` entries before the workspace owner closes.
For diagnostic execution, live RoPE tables are initialized as identity tables
with cos `1.0f` and sin `0.0f`; position-correct RoPE table population remains
part of full-serving correctness work.
With `--single-context-live-session`, token buffers, KV-cache, resident weights,
and activation workspace are allocated under one CUDA context before graph
materialization and launch-packet preflight, then closed after the preflight
evidence is recorded.
With `--run-resource-backed-smoke`, the runner prepares the generated Qwen
task-function set and launches the resource-backed DAG packets for both
serving policies while that same CUDA context is open. This is still
diagnostic: it proves scheduler completion and pointer wiring for the planned
DAG, not full Qwen numerical correctness.
Use `--resource-backed-repeat-runs` to submit fresh resource-backed graph
state repeatedly through the same prepared callable and CUDA context.
Use `--resource-backed-workload` to narrow the diagnostic to a single serving
policy, and `--resource-backed-logits-check-policy final_step` to defer the
full logits-buffer readback until the last bounded decode step.
Use `--resource-backed-numeric-task-mode unit_math` to run the resource-backed
descriptors with the safe O(n) task bodies that already have unit-math numeric
branches. RMSNorm uses an explicit external scale, and the QK-norm,
attention-output, post-attention-norm, MLP-down, and final-norm tasks use
bounded weighted elementwise branches in this mode, so this still remains
below full Qwen numerical serving correctness.
Use `--resource-backed-numeric-task-mode unit_math_full_rmsnorm` when reviewers
need to exercise the generated RMSNorm full-reduction branch instead of the
external-scale bridge. This mode is slower and still diagnostic, but it moves
the resource-backed path closer to full Qwen numerical correctness. The RMSNorm
task is block-threaded, so the generated branch reduces each full hidden vector
once per task and then writes the normalized output in a block-stride loop.
The `cuda_live_submission_descriptor_contract` maps those resource pointers
to Qwen task function ids 7100 through 7109 and records the `run_prepared`
repetition count. With `--run-submission-smoke`, it also compiles those same
function ids and launches a small controlled CUDA DAG through
`run_prepared`; this proves the descriptor task-function set is executable,
but still does not use the full Qwen resource-backed serving buffers.
The `resource_backed_graph_materialization` section checks that both serving
policies have all token fields, KV fields, and 255 resident-weight-backed DAG
task descriptors bound to concrete CUDA-live pointers.
Its launch-packet preflight also packs the host-side `CudaPersistentDagTask`
array from those pointers. When the activation workspace is live, intermediate
tasks are chained through activation buffers and the final task writes to a
float logits/sampling output buffer. QK-norm/RoPE tasks receive live
runtime-buffer pointers for their cos/sin tables through the same packet.
Activation buffers are sized from descriptor output shapes when those shapes
are available, so widened QKV and MLP intermediates no longer share the
hidden-size fallback. It still does not execute full Qwen kernels or a
full-serving decode loop.
