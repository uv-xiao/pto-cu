# NVIDIA Backend Paper-Ready Evaluation Plan: Baseline Families

## Baseline Families

The required baseline set includes:

- PTO CUDA host-schedule runtime.
- PTO CUDA persistent-device runtime.
- Direct CUDA Runtime API kernel launches.
- CUDA Driver API module launch path.
- CUDA Graph instantiate and replay.
- cuBLAS or cuBLASLt for GEMM-shaped workloads.
- CUTLASS or CuTe-based kernels for tile workloads when available.
- Triton or torch.compile for framework-generated kernels.
- Mirage Persistent Kernel, abbreviated MPK, and the baselines used in the MPK
  paper: vLLM, SGLang, FlashInfer or FlashAttention, cuBLAS or cuTLASS, CUDA,
  and Triton operator paths.
- VDCores and the baselines used in the VDCores paper: vLLM, SGLang, Mirage,
  ThunderKittens variants, and Torch plus ThunderKittens.

Local source notes already include extracted MPK and VDCores paper text under
`tmp/sources/`. Future baseline clones and command logs should stay under
`tmp/baselines/` and `tmp/cuda-backend/`.

`baseline_survey.md` records the current source state for MPK and VDCores and
the planned source-capture state for vLLM, SGLang, and ThunderKittens. The
benchmark viewer loads the same baseline readiness data from
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json`.

Paper claim readiness is tracked in
`docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix.json`.
That matrix names each claim's workloads, methods, paper baselines, hardware
targets, required metrics, current evidence, missing evidence, and promotion
gate. A claim is not paper-ready until the matrix status and raw artifacts
show complete baseline coverage.

The generated readiness audit in
`docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json` is the
human-reviewable summary of that matrix. It is produced by
`.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py` and folds
matrix gaps, paper-baseline run statuses, run-readiness statuses,
readiness-probe statuses, latest execution-attempt diagnostics, missing
viewer-result evidence, and generated next actions into one review record per
claim. The audit must stay
`not_paper_ready` until every claim has a ready matrix status and no generated
blockers.
The generated work queue in
`docs/nvidia-backend/benchmark-viewer/data/paper_readiness_work_queue.json`
flattens those next actions into one prioritized table for the HTML viewer, so
reviewers can see the remaining MPK, VDCores, vLLM, SGLang, ThunderKittens,
and PTO serving work without expanding each matrix claim.
The generated goal-progress audit in
`docs/nvidia-backend/benchmark-viewer/data/goal_progress.json` summarizes the
overall NVIDIA backend objective. It should remain `in_progress` while the
paper-grade results criterion still points at queued raw captures.
Use `.agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py`
after changing matrix, baseline, readiness, probe, result, or goal-progress
inputs so the audit, work queue, and goal-progress data move together.

Paper-baseline reproduction commands are tracked in
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json`. Those
records name the setup commands, run commands, expected tmp artifacts, required
metrics, and viewer import target for MPK, VDCores, vLLM, SGLang, and
ThunderKittens.
The persistent-device scheduler claim now has explicit MPK and VDCores run
records, `mpk_persistent_scheduler_trace` and
`vdcores_resource_policy_trace`, so reviewers can see the planned artifact
paths and required scheduler/resource-policy fields before those long runs
are captured.

Shared LLM-serving workload policies are tracked in
`docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json`. The first
two policies are `mpk_offline_decode` and `vdcores_offline_decode`, because
the MPK and VDCores papers use different decode lengths and context policies.
The MPK-comparable policy uses Qwen3-8B as the primary model, Qwen3-1.7B for
bring-up, target prompt length 64, decode length 1024, and offline batch sizes
1, 2, 4, 8, and 16. The VDCores-comparable policy also uses Qwen3-8B as the
cross-paper target through the VDCores `qwen3` schedule path, uses target
context length 128, decode length 64, and the same batch-size ladder.
Current VDCores Qwen3-8B evidence proves runtime rebuild and bounded
correctness. The full 64-token serving row is no longer only a pre-launch
capacity problem: a temporary global-instruction runtime can run `-N 64 -b 5`,
but it fails Qwen3-8B correctness thresholds, so the row remains blocked until
the global-instruction path is corrected or the schedule is segmented without
leaving the shared-instruction runtime.
The current shared-window analysis makes the segmented path concrete: under
the 512-instruction shared table, Qwen3-8B decode64 needs a lower bound of 5
compute-instruction windows and 30 memory-instruction windows per SM before it
can become a correctness-backed paper row.
The PTO full-serving gap is tracked by
`.agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py` and the
repo-owned lifecycle scaffold in
`examples/cuda/persistent_qwen_serving_scaffold.py`. The lifecycle contract in
`examples/cuda/qwen_serving_lifecycle_plan.py` now maps the shared MPK and
VDCores Qwen/Qwen3-8B serving policies to a concrete model shape, KV-cache
capacity ladder, weight-binding plan, and persistent-device callable roles.
The prompt-accounting contract in `examples/cuda/qwen_prompt_accounting.py`
now records tokenizer-observed chat-template prompt lengths for those same
serving policies.
The weight-inventory contract in `examples/cuda/qwen_weight_inventory.py` now
records the Qwen/Qwen3-8B safetensors shard count, tensor count, binding
groups, total index size from the captured index, and a config-derived
expected shape/dtype contract whose byte total matches the index.
The shard-status contract in `examples/cuda/qwen_safetensors_fetch.py` now
records the Hugging Face resolve URLs, local target paths, present/missing
counts, and resumable fetch commands for the five Qwen/Qwen3-8B safetensors
shards without downloading by default.
The metadata probe in `examples/cuda/qwen_safetensors_metadata.py` has now
opened the real local shards and validated actual shape/dtype metadata for all
399 Qwen/Qwen3-8B tensors with zero mismatches.
The CUDA binding artifact in `examples/cuda/qwen_cuda_weight_binding.py` now
maps those 399 tensors to stable binding slots, safetensors file byte ranges,
binding groups, and readonly persistent-device argument roles. On the local
A100 it also held all 16.38 GB of Qwen weights resident at once through the
existing CUDA runtime allocation/copy API, verified 16 small norm tensors by
copy-back, then freed all 399 allocations.
The persistent weight-argument artifact in
`examples/cuda/qwen_persistent_weight_args.py` maps those weight slots into 255
Qwen task descriptors that fit the current four-pointer persistent DAG
`tensor_args` ABI and cover every validated weight tensor.
The materialization artifact in
`examples/cuda/qwen_persistent_weight_materialization.py` now maps those
descriptors through the `CudaPersistentDagTask` ctypes layout. Its current
artifact is symbolic because the decode-loop runner does not yet own a live
resident pointer table, but it records the exact
`resident_weight_ptrs[slot_id]` source needed for all 399 weight arguments.
The resident weight-table artifact in
`examples/cuda/qwen_resident_weight_table.py` adds the process-scoped owner
that keeps a pointer table valid until close, feeds it to materialization, and
then frees pointers in reverse order. Its current checked artifact uses
`dry_run_pointer_lifecycle`; the decode-loop runner can now invoke the same
owner in CUDA-live mode, while actual Qwen kernel submission remains missing.
The runtime input-binding artifact in
`examples/cuda/qwen_runtime_input_binding.py` now converts tokenizer output
into padded target-length `input_ids`, matching `attention_mask`, and decode
`output_ids` buffer descriptors for the MPK and VDCores serving policies.
The CUDA token-buffer binding artifact in
`examples/cuda/qwen_cuda_token_buffer_binding.py` now allocates those six
paper-policy token buffers with the CUDA host runtime, copies host token data
to device memory, verifies copy-back, and frees the temporary owner scope.
The token pointer-table artifact in
`examples/cuda/qwen_token_pointer_table.py` now keeps six token pointers live
while persistent decode arguments are materialized. Its checked artifact uses
`dry_run_pointer_lifecycle`; `--cuda-live` remains the decode-loop runner mode
needed for actual Qwen kernel submission.
The KV-cache binding artifact in
`examples/cuda/qwen_kv_cache_binding.py` now splits the planned cache into key
and value buffers and maps them to persistent DAG `c` and `d` fields. Its
checked artifact covers 20 CUDA-live pointers totaling 5.45 GiB across the
MPK and VDCores batch ladders. It allocates the planned KV-cache memory but
does not prefill Qwen attention KV values.
The decode-loop runner artifact in
`examples/cuda/qwen_decode_loop_runner.py` now composes the token pointer,
KV-cache, and resident weight lifecycles into a persistent DAG submission
order. It records 1088 planned decode iterations across the two serving
policies, makes output-token accounting reviewable, and records the
diagnostic bridge from owner-owned fields into the repeated proxy live runner.
It can also execute the repeated unit-math diagnostic from the runner entry
point; the current bridge artifact is
`tmp/cuda-backend/pto-serving-decode-loop-unit-math-bridge-2026-06-01/qwen-decode-loop-runner.json`.
It can also open the token pointer table in CUDA-live mode from the runner;
the current partial resource-owner artifact is
`tmp/cuda-backend/pto-serving-decode-loop-token-kv-resident-live-2026-06-02/qwen-decode-loop-runner.json`.
It now opens token, KV-cache, and resident-weight owners in CUDA-live mode.
The current descriptor artifact is
`tmp/cuda-backend/pto-serving-decode-loop-submission-descriptors-2026-06-02/qwen-decode-loop-runner.json`;
it records resource-backed Qwen function ids 7100 through 7109, graph task
counts, and `run_prepared` repetition counts for the MPK and VDCores serving
policies. The runner can now attach a diagnostic descriptor smoke that
compiles those same function ids and executes a small controlled
`cuda/persistent_device` DAG, but the full resource-backed Qwen decode loop
still remains unexecuted. The current live-owner smoke artifact is
`tmp/cuda-backend/pto-serving-decode-loop-submission-smoke-live-2026-06-02/`
`qwen-decode-loop-runner.json`, and the benchmark viewer imports it as
`serving_coverage=diagnostic_qwen_descriptor_smoke`.
The current resource-backed logits-reference artifact is
`tmp/cuda-backend/pto-serving-resource-backed-full-logits-check-2026-06-02/qwen-decode-loop-runner.json`.
It reuses one prepared callable inside the single CUDA context for three
resource-backed submissions per serving policy, completing `765` diagnostic
tasks per policy with zero scheduler errors. The final diagnostic logits task
now writes all `2,430,976` logits-buffer elements per serving policy and
records `full_logits_buffer_checked` coverage. It checks all 2,430,976
written logits elements against the diagnostic formula
`out[i]=hidden[i%hidden_elements]*lm_head[i&3]`, now passing with zero
mismatches. The prior mismatch came from generated task-body snippets that
returned from inside the persistent DAG grid-stride wrapper before all logits
elements were written. This is still diagnostic execution, not full Qwen
numerical correctness or a full-serving row.
The task-body source artifact in
`examples/cuda/qwen_persistent_task_bodies.py` now renders Qwen persistent
task bodies through the existing persistent DAG source generator. It records
source-level consumption of token fields `a`, `b`, and `out`, KV-cache fields
`c` and `d`, and weight `tensor_args`. This is source-generation evidence,
not a numerically correct Qwen kernel implementation. The persistent DAG ABI
now exposes mutable `c` and `d` fields, so the artifact also records
KV-cache writeback field access. It also includes a controlled proxy numeric
oracle for the current deterministic scaffold formulas; that oracle is not
full Qwen correctness evidence.
It also carries a small Qwen unit math oracle covering RMSNorm, projection,
single-token attention cache writeback, SiLU/SwiGLU, and logits equations;
the generated CUDA source contains a matching opt-in unit-math path, and the
current A100 artifact executes that path through `cuda_live` with zero
observed error.
The persistent decode-argument artifact in
`examples/cuda/qwen_persistent_decode_args.py` maps those token buffers onto
the persistent DAG `a`, `b`, and `out` fields while preserving `tensor_args`
for Qwen weights.
The current raw artifacts are
`tmp/cuda-backend/pto-serving-lifecycle-b95ff321/qwen-serving-lifecycle-plan.json`,
`tmp/cuda-backend/pto-serving-tokenizer-b95ff321/qwen-prompt-accounting.json`,
`tmp/cuda-backend/pto-serving-input-binding-2026-06-01/qwen-runtime-input-binding.json`,
`tmp/cuda-backend/pto-serving-token-buffer-2026-06-01/qwen-cuda-token-buffer-binding.json`,
`tmp/cuda-backend/pto-serving-decode-args-2026-06-01/qwen-persistent-decode-args.json`,
`tmp/cuda-backend/pto-serving-token-pointers-2026-06-01/qwen-token-pointer-table.json`,
`tmp/cuda-backend/pto-serving-kv-cache-2026-06-01/qwen-kv-cache-binding.json`,
`tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/qwen-decode-loop-runner.json`,
`tmp/cuda-backend/pto-serving-decode-loop-bridge-2026-06-01/qwen-decode-loop-runner.json`,
`tmp/cuda-backend/pto-serving-task-bodies-2026-06-01/qwen-persistent-task-bodies.json`,
`tmp/cuda-backend/pto-serving-task-bodies-qwen-unit-2026-06-01/qwen-persistent-task-bodies.json`,
`tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/qwen-unit-math-live.json`,
`tmp/cuda-backend/pto-serving-proxy-live-2026-06-01/qwen-proxy-live.json`,
`tmp/cuda-backend/pto-serving-microdecode-live-2026-06-01/qwen-microdecode-live.json`,
`tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/qwen-microdecode-loop.json`,
`tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json`,
`tmp/cuda-backend/pto-serving-shards-a16851f6/qwen-safetensors-shards.json`,
`tmp/cuda-backend/pto-serving-safetensors-a16851f6/qwen-safetensors-metadata.json`,
`tmp/cuda-backend/pto-serving-weight-residency-1ae913c9/qwen-cuda-weight-residency.json`,
`tmp/cuda-backend/pto-serving-weight-args-21589e81/qwen-persistent-weight-args.json`,
`tmp/cuda-backend/pto-serving-weight-materialization-2026-06-01/qwen-persistent-weight-materialization.json`,
`tmp/cuda-backend/pto-serving-resident-weight-table-2026-06-01/qwen-resident-weight-table.json`,
`tmp/cuda-backend/pto-serving-scaffold-2026-06-01/qwen-serving-scaffold.json`,
and
`tmp/cuda-backend/pto-serving-preflight-2026-06-01/pto-serving-preflight.json`.
They prove the current viewer has controlled attention-tile proxy, repeated
diagnostic unit-math, and diagnostic microdecode loop rows for PTO
serving-equivalent evidence, plus a partial runtime plan for the
Qwen3-8B KV-cache and task mapping, tokenizer-observed prompt counts, a
host-side runtime token-buffer plan with padded target-length `input_ids` and
`attention_mask`, CUDA token-buffer allocation/copy-back verification,
persistent decode token argument binding through `a`, `b`, and `out`,
dry-run KV-cache key/value binding through `c` and `d`,
dry-run decode-loop owner ordering and DAG submission planning,
source-level generated Qwen task bodies that consume token, KV-cache, and
weight argument fields through the existing persistent DAG source generator,
one controlled QKV proxy task launched live through `cuda/persistent_device`
on A100 with zero scheduler errors and exact copied-back `out`/`c`/`d` values,
one controlled QKV-to-logits proxy DAG launched live through
`cuda/persistent_device` with dependency release across three tasks,
one controlled unit-math DAG reused across three prepared submissions with
12 completed task executions,
one controlled repeated proxy decode loop that reuses a prepared callable
across three `run_prepared` submissions while carrying mutable KV state,
safetensors shard/tensor inventory, and the expected weight shape/dtype
contract. It also proves local Qwen shard placement and actual safetensors
shape/dtype validation for 399 tensors across five shards, plus full CUDA
weight residency for all 399 tensors in a probe process and a persistent DAG
weight-argument manifest that fits the current ABI. It also proves a
ctypes-backed materialization plan for the resident weight pointer table and a
dry-run owner lifecycle that binds 399 pointers and frees them after
materialization. The repo-owned PTO CUDA path still lacks numerically correct
Qwen kernel bodies, full `cuda_live` decode-loop execution, and full-serving
`viewer_result_import`, so no PTO `Qwen/Qwen3-8B` full-serving row can be
imported yet.
The PTO preflight checks the promotion gate per row: the row must be
`llm_serving_decode` / `pto_persistent_device`, name `Qwen/Qwen3-8B`, use
`statistic.serving_coverage=full_serving`, pass correctness, include the MPK
or VDCores serving workload ID, and carry latency plus throughput metrics.
Both `mpk_offline_decode` and `vdcores_offline_decode` rows are required before
the PTO full-serving evidence item can pass.
Every serving baseline run record must reference one of these policy IDs and
require both `model_and_prompt_shape` and `batch_or_concurrency_policy` before
it can be imported. Imported rows must record actual tokenizer counts, model
identity, decode count, and batch size in raw JSON.
Every imported `llm_serving_decode` result must also record
`statistic.serving_coverage`. `full_serving` and
`full_serving_latency_caveat` are the only coverage classes that can support a
full-serving paper comparison. `controlled_attention_tile_proxy`,
`diagnostic_unit_math`, `diagnostic_microdecode`, and `native_bringup` rows
remain useful evidence, but they cannot close the PTO, VDCores, or
ThunderKittens full-serving gaps.
