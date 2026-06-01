# NVIDIA Backend Paper-Ready Evaluation Plan

## Goal

The evaluation must be paper-ready: a reviewer should be able to understand the
claim, reproduce the command, inspect raw artifacts, and compare PTO CUDA
against credible NVIDIA baselines.

The plan treats current smoke results as early evidence, not final paper
results.

## Review Map

- [Baseline families](evaluation_plan/baseline_families.md): PTO CUDA,
  direct CUDA, CUDA Graph, cuBLAS or cuBLASLt, CUTLASS or CuTe, Triton,
  Mirage Persistent Kernel, VDCores, and the MPK/VDCores paper baselines.
- [Workloads](evaluation_plan/workloads.md): vector ABI, DAG scheduling,
  tensor, lifecycle, and LLM-serving workload ladders.
- [Hardware matrix](evaluation_plan/hardware_matrix.md): A100/H200 minimum
  development targets and optional H100/B200 extensions.
- [Metrics](evaluation_plan/metrics.md): correctness, latency, launch
  overhead, scheduler overhead, throughput, distributions, and resource
  policy fields.
- [Reproducibility rules](evaluation_plan/reproducibility.md): required raw
  artifact metadata, viewer import paths, source captures, and remote H200
  refresh/fallback policy.
- [Paper outputs](evaluation_plan/paper_outputs.md): target tables, figures,
  and appendix artifacts.
- [Dispatcher backlog](evaluation_plan/dispatcher_backlog.md): first concrete
  execution backlog for paper-ready evidence collection.

## Required Baselines

The required baseline set explicitly includes Mirage Persistent Kernel, VDCores,
CUDA Graph, cuBLAS or cuBLASLt, CUTLASS or CuTe, Triton, vLLM, SGLang,
ThunderKittens, direct CUDA Runtime API launches, and CUDA Driver API launches
on A100 and H200 where applicable.

## Generated Review Data

The benchmark viewer loads committed JSON data under
`docs/nvidia-backend/benchmark-viewer/data/`. Paper claim readiness is tracked
by the paper evaluation matrix, readiness audit, readiness work queue, and goal
progress data. The capture-import mapping now uses a sharded
`capture_imports/` collection instead of one long JSON file.
