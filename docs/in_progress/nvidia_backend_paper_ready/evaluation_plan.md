# NVIDIA Backend Paper-Ready Evaluation Plan

## Goal

The evaluation must be paper-ready: a reviewer should be able to understand the
claim, reproduce the command, inspect raw artifacts, and compare PTO CUDA
against credible NVIDIA baselines.

The plan treats current smoke results as early evidence, not final paper
results.

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
The PTO full-serving gap is tracked by
`.agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py` and the
repo-owned lifecycle scaffold in
`examples/cuda/persistent_qwen_serving_scaffold.py`. The current raw artifacts
are
`tmp/cuda-backend/pto-serving-scaffold-76d4fca4/qwen-serving-scaffold.json`
and
`tmp/cuda-backend/pto-serving-preflight-76d4fca4/pto-serving-preflight.json`.
They prove the current viewer has only the controlled attention-tile proxy row
for PTO serving-equivalent evidence. The repo-owned PTO CUDA path still lacks
the scaffold stages `qwen_tokenizer`, `qwen_weight_loader`,
`kv_cache_lifecycle`, `decode_loop_runner`, and `viewer_result_import`, so no
PTO `Qwen/Qwen3-8B` full-serving row can be imported yet.
Every serving baseline run record must reference one of these policy IDs and
require both `model_and_prompt_shape` and `batch_or_concurrency_policy` before
it can be imported. Imported rows must record actual tokenizer counts, model
identity, decode count, and batch size in raw JSON.
Every imported `llm_serving_decode` result must also record
`statistic.serving_coverage`. `full_serving` and
`full_serving_latency_caveat` are the only coverage classes that can support a
full-serving paper comparison. `controlled_attention_tile_proxy`,
`diagnostic_microdecode`, and `native_bringup` rows remain useful evidence,
but they cannot close the PTO, VDCores, or ThunderKittens full-serving gaps.

## Workloads

The workload ladder should grow from controlled kernels to paper-level systems:

- vector ABI workloads: add, mul, scale, square, axpy, affine, triad, quad, and
  generic argument packets;
- DAG scheduling workloads: chain, diamond, layered-cross, fan-in, fan-out, and
  queue-capacity pressure;
- tensor workloads: tiled GEMM, tensor-core tile kernels, shape sweeps, and
  stream or graph variants;
- lifecycle workloads: repeated runtime init, module load, allocation, copy,
  launch, synchronize, teardown, and rebuild separation;
- LLM-serving workloads: decode microsteps, paged KV-cache movement, attention,
  GEMM, all-reduce or all-gather when multi-GPU is in scope.

Each workload must have a benchmark viewer entry, example or script entry point,
and a raw artifact path.

## Hardware Matrix

Minimum paper-development hardware:

- A100 for local development and compatibility with current captures.
- H200 for Hopper-class scheduling, tensor-core, and remote evaluation.

Optional paper extensions:

- H100 when available for direct comparison with MPK and VDCores paper
  hardware.
- B200 or Blackwell-class hardware only after the A100/H200 matrix is stable.

Every result records GPU model, CUDA toolkit, driver version, compute target,
clock policy when known, and whether Multi-Process Service or exclusive mode
was active.

## Metrics

Collect at least:

- correctness status and checksum or reference comparison;
- end-to-end latency;
- device-only elapsed time;
- host launch overhead;
- scheduler overhead;
- throughput;
- p50, p90, p99, mean, standard deviation, min, max, and sample count;
- occupancy or resource policy where available;
- stream count, graph node count, scheduler blocks, worker blocks, block
  dimension, and queue capacity for PTO runtimes.

Paper figures should separate launch overhead, device execution, and scheduler
overhead so CUDA Graph, host-schedule, and persistent-device claims are not
collapsed into one number.

## Reproducibility Rules

Each benchmark command writes JSON under `tmp/cuda-backend/` and can be indexed
into the viewer data. A paper result is not accepted unless it records:

- command;
- pto-cu commit;
- source baseline commit or release;
- hardware;
- CUDA toolkit and driver;
- input shape;
- repeat count and warmup count;
- raw output path;
- validation command.

Current PTO microbenchmark captures can be converted to viewer result records
with `.agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py`. The
script uses `docs/nvidia-backend/benchmark-viewer/data/capture_imports.json`
to map raw capture baselines onto viewer benchmark and method IDs.
The host-launch A100 capture at
`tmp/cuda-backend/host-launch-a100-8b6cdaee/` now imports its 10-repeat
`direct_driver` and `direct_driver_graph` rows into `results.json`, giving the
host-schedule launch claim explicit raw CUDA Driver launch and CUDA Driver
graph evidence on A100 while H200 Driver rows, true Runtime API rows, and
tensor-shape graph rows remain open.
The follow-up A100 capture at
`tmp/cuda-backend/host-launch-runtime-a100-e429c07b/` imports a 10-repeat
`direct_runtime` row produced by an nvcc-built shared library that calls
`cudaLaunchKernel`.
The H200 host-launch capture at
`tmp/cuda-backend/host-launch-h200-ec8f272e/` imports 10-repeat
`direct_runtime`, `direct_driver`, and `direct_driver_graph` rows for the same
`n=1024` vector shape. This closes the cross-GPU vector-launch comparison gap
for Runtime API, Driver API, and Driver graph paths.
`cuda_viewer_export.py` now imports p50, p90, p99, mean, standard deviation,
minimum, and maximum host/device latency fields for repeated raw captures, so
the current 10-repeat A100/H200 host-launch rows expose distribution shape in
the viewer instead of selected medians alone.
The selected tensor-launch captures at
`tmp/cuda-backend/tensor-launch-a100-09462d04/` and
`tmp/cuda-backend/tensor-launch-h200-09462d04/` import 10-repeat
`direct_runtime`, `direct_driver`, and `direct_driver_graph` rows for the
`n=1024`, `16x16x16` naive SGEMM tensor shape. This closes the selected
tensor-launch comparison gap for the host-schedule launch claim. Stream-count
or graph-replay sweep distributions remain open.

Paper-baseline raw captures can be converted with
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py`.
The script reads `paper_baseline_runs.json`, maps each raw
`paper_baseline_run_id` to the corresponding paper-baseline method, and emits
viewer `result_records` for MPK, VDCores, vLLM, SGLang, or ThunderKittens.
This keeps paper-baseline rows generated from raw artifacts instead of
hand-edited tables.
After a capture is accepted for committed viewer evidence,
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py`
updates the viewer results, marks the matching run imported, and regenerates
the paper-readiness audit from the same inputs.
Before executing planned MPK, VDCores, vLLM, or SGLang paper-baseline runs,
refresh `paper_baseline_run_readiness.json` with
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`.
The viewer then shows whether model access, entrypoints, reproduction-command
contracts, expected artifacts, paired probe status, and the VDCores
`dae.runtime` build are ready. Readiness rows are pre-run evidence only; the
paper claim remains blocked until measured raw JSON is imported.

Serving baseline commands can be materialized with
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`.
The script reads `serving_workloads.json` plus `paper_baseline_runs.json` and
emits one command-plan row per baseline, policy, and batch size. The current
primary-model command plan is committed as
`docs/nvidia-backend/benchmark-viewer/data/serving_command_plan.json` for
human review and is also regenerated under `tmp/cuda-backend/` before long
runs. The current command planner emits 35 rows covering MPK, VDCores, vLLM,
SGLang, and the ThunderKittens decode-attention serving-equivalent over the
MPK/VDCores policy batch ladders. This is not a performance result; it is the
reproducible launch contract that long H200 runs must execute before their raw
JSON can be imported into the viewer. SGLang launch rows explicitly prepend the
pinned source checkout to `PYTHONPATH`, so generated commands do not
accidentally use a globally installed SGLang package. ThunderKittens rows use
the bounded MHA capture wrapper as a controlled serving-family kernel baseline
for the VDCores decode policy.

The viewer also includes one PTO `persistent_device` controlled
serving-equivalent row for `vdcores_offline_decode`: a H200 tensor-core
persistent DAG capture mapped to batch 4, 128 prompt tokens, and 64 decode
tokens. This proves the PTO side has a reviewable serving-policy-shaped row,
but it is still only an attention-tile proxy. Full MPK, VDCores, vLLM, SGLang,
ThunderKittens-family, and end-to-end PTO serving artifacts remain required
before the LLM serving claim can be paper-ready.

Before full baseline builds, paper-baseline readiness probes can be captured
with `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`. The
probe checks pinned source commits, selected entrypoint paths, Python syntax,
required Python modules, CUDA toolkit availability, and visible GPUs. The
latest paired readiness artifact is recorded at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`.
Use `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py`
for paired A100/H200 readiness. In `--sync-remote-tree` mode it copies the
local checkout to the H200 host, skips remote Git, runs the remote probe with
the explicit CUDA toolkit path, and copies `h200-probe.json` back beside the
local `a100-probe.json`. The paired runner also syncs `tmp/baselines/`
separately, because those source checkouts are required probe inputs but
generated `tmp/cuda-backend/` outputs should not be copied as part of the repo
tree sync.
The 43b927ed paired probe checks the `transformers` module for MPK and
VDCores, matching the selected Qwen3 and Llama entrypoint imports. It also
checks that the selected SGLang benchmark modules import from the pinned
source checkout, not only that the source files exist. The H200 project venv
has `transformers` installed, so MPK, VDCores, and ThunderKittens remain
setup-ready on H200 while vLLM and SGLang remain partial. vLLM now has a
reviewed Python 3.10 source-overlay path that copies the pinned checkout under
`tmp/cuda-backend/paper-baselines/source-overlays/`, unsets
`Py_LIMITED_API` only for the copied spinloop CXX target, derives the package
version from the pinned upstream checkout, builds editable vLLM from the
overlay, and passes the selected local validation imports. SGLang is partial
because H200 is missing `orjson`, while the local A100 import path currently
hits a torch/torchvision operator-registration mismatch.
ThunderKittens readiness must include the selected PyTorch-extension
dependencies (`torch`, `pybind11`, `numpy`, `pandas`, `matplotlib`, and
`tqdm`), not just source-file existence. After installing those modules in
the H200 project venv, the current H200 probe is ready for the selected
ThunderKittens setup path. The selected official H100 MHA benchmark now has
an FA3-enabled H200 run as well: FlashAttention-3 was built from the
`tmp/baselines/flash-attention/hopper` source clone with a narrowed SM90 BF16
head-dim-128 build, and the unmodified ThunderKittens benchmark was run with a
local compatibility shim that requests `return_attn_probs=True` so the current
FA3 API returns the `(out, lse)` tuple expected by ThunderKittens. The FA3
rows completed for forward/backward, causal/non-causal, and sequence lengths
768, 1536, 3072, 6144, and 12288. A follow-up isolated PyTorch reference
capture ran each selected large reference cell in a fresh H200 process with
expandable allocator segments. It recovered every 6144-token cell and left only
12288-token dense PyTorch reference cells OOM, so the remaining official-sweep
capacity issue is recorded as an accepted evidence-policy exception instead of
a tensor-core blocker. The exception is scoped only to the infeasible dense
PyTorch reference rows; paper tables may footnote those rows as
OOM/not-applicable and must not impute performance or correctness numbers from
them.
The LLM-serving claim also has a planned ThunderKittens
`thunderkittens_decode_attention_tile` run record. Its readiness is generated
from the selected source tree, repo-local capture wrapper, expected tmp
artifact paths, required metrics, and paired ThunderKittens probe status, so
the audit no longer hides ThunderKittens behind a missing-run blocker. The
current H200 capture imports five VDCores-policy batch rows. The H100 MHA
wrapper pads the 64-token decode policy to `n=256` so the kernel launch grid
is nonzero; the raw and viewer rows still preserve `prompt_tokens=128` and
`decode_tokens=64` for the serving comparison.
The first bounded ThunderKittens MHA capture is recorded under
`tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/`. It
uses `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py`
to run two H200 causal BF16 MHA shapes with five warmups and twenty timed
CUDA-event repeats, compare against PyTorch scaled-dot-product attention, and
export viewer-compatible paper-baseline records. This promotes the selected
ThunderKittens bounded-capture row from setup-ready to imported viewer
evidence. The full upstream correctness and benchmark sweeps are tracked by
the separate `thunderkittens_full_sweep` run. Its expected artifacts now
include the bounded repo-owned sweep, the original official benchmark probe,
and the FA3-enabled official rerun under
`tmp/cuda-backend/paper-baselines/thunderkittens/upstream-benchmark-fa3-7371626c/`.
Imported paper-baseline run records are not allowed to keep missing future
artifacts in their `expected_artifacts` lists.

Remote H200 runs should prefer Git refresh when available and use SSH
tree-sync fallback when remote Git fails. The selected path is part of the
artifact metadata.

## Paper Tables And Figures

Target review artifacts:

- table of workload definitions and input shapes;
- table of methods and launch models;
- A100 latency and throughput table;
- H200 latency and throughput table;
- scheduler-overhead breakdown for persistent-device;
- CUDA stream and graph concurrency comparison for host-schedule;
- baseline comparison against MPK, VDCores, and their paper baselines;
- ablation table for stream count, scheduler blocks, worker blocks, queue
  capacity, block dimension, and graph shape;
- reproducibility appendix with commands and artifact paths.

## First Dispatcher Backlog

1. Audit current viewer data against the shared schema.
2. Add missing fields for method category, launch model, hardware, and raw
   artifact paths.
3. Create a baseline source index under `tmp/` for MPK, VDCores, and paper
   baselines.
4. Clone or inspect MPK and VDCores repositories under `tmp/baselines/`.
5. Use `serving_workloads.json` to run MPK, VDCores, vLLM, and SGLang with
   comparable model, prompt, decode, and batch-size policies.
6. Extend viewer export scripts as new benchmark families and paper baselines
   produce raw JSON.
7. Run A100 and H200 paired captures for the current PTO workloads.
8. Add controlled CUDA Graph, cuBLAS, and direct launch baselines.
