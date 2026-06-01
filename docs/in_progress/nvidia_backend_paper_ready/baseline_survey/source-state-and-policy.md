# Baseline Source State And Serving Policy

## Purpose

This survey turns the paper-baseline requirement into reviewable source state.
It records which external systems have been inspected, where their local source
copies live under `tmp/`, and what each child evaluation slice must reproduce.

## Local Source State

| System | Upstream | Local source | Commit | Status |
| --- | --- | --- | --- | --- |
| MPK | `mirage-project/mirage` branch `mpk` | `tmp/baselines/mirage-mpk` | `bde2dec1736d612f7a2e4c89e6182560a863072f` | cloned for survey |
| VDCores | `vdcores/vdcores` branch `main` | `tmp/baselines/vdcores` | `5247328cf3f893ed9df95f9f38e7e9a97f0cbfb1` | cloned for survey |
| vLLM | `vllm-project/vllm` branch `main` | `tmp/baselines/vllm` | `27fa5aa3b952a6108de127423397e50364a95fcb` | cloned for survey |
| SGLang | `sgl-project/sglang` branch `main` | `tmp/baselines/sglang` | `7ed53d15f357ea4d722c1980c2cb35e8367d8bb0` | cloned for survey |
| ThunderKittens | `HazyResearch/ThunderKittens` branch `main` | `tmp/baselines/thunderkittens` | `34b15f7e7012de25ae162c8d9dc85296dd342676` | cloned for survey |

Comparator dependency sources used by those baseline runs are also kept under
`tmp/` for review. The current FlashAttention source clone is
`tmp/baselines/flash-attention` at commit
`6dba0373b775196039aedda01cd14c51662965d8`. It was used to build the
FlashAttention-3 Hopper module required by the official ThunderKittens H100
MHA benchmark.

The committed viewer data mirrors this table in
`docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json` so the
human-reviewable benchmark viewer can show baseline readiness without relying
on private terminal history. Reproduction commands for these systems live in
`docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json` so the
viewer can show which setup/run commands and tmp artifacts are expected before
a baseline can be imported as paper evidence.
Serving workload policies live in
`docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json`. The MPK
and VDCores papers use different context and decode lengths, so the current
survey records two comparable policy IDs instead of pretending there is one
universal serving row:

- `mpk_offline_decode`: Qwen3-8B primary, prompt target 64, decode 1024,
  offline batch sizes 1, 2, 4, 8, and 16.
- `vdcores_offline_decode`: Qwen3-8B cross-paper target through the VDCores
  `qwen3` schedule path, context target 128, decode 64, offline batch sizes
  1, 2, 4, 8, and 16.

The current primary-model launch plan for those policies is materialized at
`tmp/cuda-backend/paper-baselines/serving-runs/plan-43b927ed.json` by
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`.
It expands the two policy IDs into MPK, VDCores, vLLM, and SGLang command rows
for each batch size, including the raw artifact paths expected by the viewer
importer after the long baseline runs complete. SGLang commands in the plan
prefix `PYTHONPATH=$PWD/tmp/baselines/sglang/python:$PYTHONPATH` so benchmark
modules resolve from the pinned source checkout.

The persistent-device scheduler comparison has separate run contracts in
`paper_baseline_runs.json`: `mpk_persistent_scheduler_trace` and
`vdcores_resource_policy_trace`. These are not serving throughput rows; they
name the planned MPK and VDCores artifacts needed to compare generated
persistent kernels, virtual-core queue/resource policy, dispatch traces, and
scheduler overhead against PTO CUDA persistent-device captures.
