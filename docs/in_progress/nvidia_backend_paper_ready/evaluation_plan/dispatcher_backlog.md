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

Active remaining-gap pages:

- `status/remaining-gaps/qwen-full-serving-correctness.md`: close full Qwen
  numerical correctness before importing PTO full-serving rows.
- `status/remaining-gaps/tuned-tensor-workloads.md`: capture multi-repeat
  model-shape tensor workload rows for PTO and comparable baselines.

The active work below is paper-readiness capture, baseline correctness, and
result-import work.

## Active Paper Work Items

The generated paper-readiness queue currently has `4` active paper work items:

- `paper_readiness_work_item_001`: implement and import full-serving PTO
  persistent-device Qwen/Qwen3-8B rows for MPK and VDCores policies.
  Evidence context: Lifecycle and input artifacts:
  persistent_qwen_serving_scaffold, qwen_serving_lifecycle_plan,
  qwen_prompt_accounting, qwen_runtime_input_binding,
  qwen_cuda_token_buffer_binding, qwen_persistent_decode_args, and
  qwen_kv_cache_binding.
  Command-plan selectors:
  `pto_persistent_device_qwen3_8b_full_serving:mpk_offline_decode`,
  `pto_persistent_device_qwen3_8b_full_serving:vdcores_offline_decode`.
- `paper_readiness_work_item_002`: run and import the VDCores full-serving
  Qwen/Qwen3-8B row after correctness is fixed for the shared-instruction path.
  Evidence context: VDCores Qwen3-8B is tracked by the planned-not-run
  vdcores_qwen3_8b_decode_preflight row for the vdcores_offline_decode policy.
  Command-plan selector:
  `vdcores_qwen3_8b_decode_preflight:vdcores_offline_decode`.
- `paper_readiness_work_item_003`: import ThunderKittens-family full-serving
  Qwen/Qwen3-8B rows beyond the current attention-tile proxy.
  Evidence context: ThunderKittens decode-attention tile rows are imported for
  the VDCores offline-decode batch ladder as controlled attention-tile proxy
  evidence.
  Command-plan selector:
  `thunderkittens_decode_attention_tile:vdcores_offline_decode`.
- `paper_readiness_work_item_004`: resolve the VDCores shared-instruction
  window plan into a runnable baseline before importing the preflight row.
  Evidence context: The latest VDCores shared-instruction attempt generated
  analysis evidence for the Qwen3-8B decode64 instruction footprint.
  Command-plan selector:
  `vdcores_qwen3_8b_decode_preflight:vdcores_offline_decode`.

## Promotion Rules

- Keep backend implementation closure tied to
  `docs/nvidia-backend/status.md`; linked remaining-gap pages make that
  criterion active again.
- Do not mark paper results complete while
  `paper_readiness_work_queue.json` contains active work items.
- Promote a paper row only after raw artifacts under `tmp/` import into viewer
  records with matching model, prompt, decode, batch policy, correctness,
  latency or throughput statistics, and source provenance.
