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

Paper-baseline reproduction commands are tracked in
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json`. Those
records name the setup commands, run commands, expected tmp artifacts, required
metrics, and viewer import target for MPK, VDCores, vLLM, SGLang, and
ThunderKittens.

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

Paper-baseline raw captures can be converted with
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py`.
The script reads `paper_baseline_runs.json`, maps each raw
`paper_baseline_run_id` to the corresponding paper-baseline method, and emits
viewer `result_records` for MPK, VDCores, vLLM, SGLang, or ThunderKittens.
This keeps paper-baseline rows generated from raw artifacts instead of
hand-edited tables.

Before full baseline builds, paper-baseline readiness probes can be captured
with `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`. The
probe checks pinned source commits, selected entrypoint paths, Python syntax,
required Python modules, CUDA toolkit availability, and visible GPUs. The
latest paired readiness artifact is recorded at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-67c5c655/`.
Use `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py`
for paired A100/H200 readiness. In `--sync-remote-tree` mode it copies the
local checkout to the H200 host, skips remote Git, runs the remote probe with
the explicit CUDA toolkit path, and copies `h200-probe.json` back beside the
local `a100-probe.json`. The paired runner also syncs `tmp/baselines/`
separately, because those source checkouts are required probe inputs but
generated `tmp/cuda-backend/` outputs should not be copied as part of the repo
tree sync.
ThunderKittens readiness must include the selected PyTorch-extension
dependencies (`torch`, `pybind11`, `numpy`, `pandas`, `matplotlib`, and
`tqdm`), not just source-file existence. After installing those modules in
the H200 project venv, the current H200 probe is ready for the selected
ThunderKittens setup path, but full correctness and benchmark sweeps remain
future paper-evaluation work.

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
5. Extend viewer export scripts as new benchmark families and paper baselines
   produce raw JSON.
6. Run A100 and H200 paired captures for the current PTO workloads.
7. Add controlled CUDA Graph, cuBLAS, and direct launch baselines.
8. Decide which LLM-serving workload is the first credible paper target.
