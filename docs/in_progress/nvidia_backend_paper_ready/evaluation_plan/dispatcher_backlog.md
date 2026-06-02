# NVIDIA Backend Paper-Ready Evaluation Plan: Dispatcher Backlog

## Completed First Pass

- Viewer schema and sharded review data are guarded by
  `.agents/checks/validate_benchmark_viewer_data.py`.
- Method category, launch model, hardware, statistic, correctness, and raw
  artifact paths are required by the shared contract and viewer validators.
- Baseline source checkouts for MPK, VDCores, vLLM, SGLang, ThunderKittens,
  CUTLASS, and FlashAttention live under `tmp/baselines/` for local review.
- Serving command plans, paired A100/H200 probes, CUDA capture imports, and
  paper-baseline result import scripts exist as committed review workflows.
- Current A100/H200 host-launch, tensor-launch, persistent-device diagnostic,
  framework-baseline, and paper-baseline rows are imported into the viewer.

## Active Backend Gaps

- `status/remaining-gaps/persistent-scheduler-generalization/index.md`:
  persistent-device scheduling still needs general graph coverage beyond the
  verified tracer-bullet shapes.
- `status/remaining-gaps/tuned-tensor-workloads.md`: tensor rows still need
  tuned PTO tensor bodies and multi-repeat A100/H200 paper captures.

## Active Paper Work Items

The generated paper-readiness queue currently has `4` active paper work items:

- `paper_readiness_work_item_001`: implement and import full-serving PTO
  persistent-device Qwen/Qwen3-8B rows for MPK and VDCores policies.
- `paper_readiness_work_item_002`: run and import the VDCores full-serving
  Qwen/Qwen3-8B row after correctness is fixed for the shared-instruction path.
  Command-plan selector:
  `vdcores_qwen3_8b_decode_preflight:vdcores_offline_decode`.
- `paper_readiness_work_item_003`: import ThunderKittens-family full-serving
  Qwen/Qwen3-8B rows beyond the current attention-tile proxy.
  Command-plan selector:
  `thunderkittens_decode_attention_tile:vdcores_offline_decode`.
- `paper_readiness_work_item_004`: resolve the VDCores shared-instruction
  window plan into a runnable baseline before importing the preflight row.

## Promotion Rules

- Do not mark backend implementation complete while `docs/nvidia-backend/status.md`
  links any remaining-gap page.
- Do not mark paper results complete while
  `paper_readiness_work_queue.json` contains active work items.
- Promote a paper row only after raw artifacts under `tmp/` import into viewer
  records with matching model, prompt, decode, batch policy, correctness,
  latency or throughput statistics, and source provenance.
