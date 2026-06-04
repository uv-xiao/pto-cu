# CUDA Backend Status: Qwen Full-Serving Correctness

## Open Gap

Qwen full-serving paper-readiness remains open. The resource-backed
Qwen/Qwen3-8B diagnostic runs execute generated persistent-device task bodies
with real tokenizer, resident safetensors, live activation buffers, KV cache,
and device token feedback, but they still do not match the Hugging Face
full-model reference token/logit path.

Recent implementation work narrows the gap by making `qwen_attention_o`
compute head-level dot-product decode attention, matching Qwen split-half
`rotate_half` RoPE, and exposing one projection-window policy for QKV,
attention-output, and MLP projection tasks. The attention-output task now
caches bounded attention values before applying `o_proj_weight`, which makes
full first-layer projection execution practical. The MLP down-projection now
adds the down projection to the full pre-MLP residual stream by combining the
attention-output residual with the original layer input. This is still not
enough to promote PTO rows to full-serving correctness.

The QK-norm and attention-output task descriptors now keep launch-packet
`rows` as workload batch rows instead of overloading it with query-head count;
the generated kernels derive query-head count from width/head metadata.

The first non-finite layer-prefix row has been localized and fixed for the
one-layer MPK path. `qwen_mlp_down` was binding `tensor_args[1]` to token IDs
when `_mlp_down` descriptors failed to resolve their layer input norm prefix;
that made the down-projection residual read integer token storage as floats.
The fixed launch packet binds the embedding activation for layer 0. The
post-fix one-layer A100 artifact reports no row-0 non-finite activations, full
finite logits, populated top-k, and a passing diagnostic logits reference.
The same resource-backed MPK path now scales through the full 36-layer
decoder prefix with full projection and logits windows: 255 tasks complete
with zero scheduler errors, no row-0 non-finite activations, full finite
logits, populated top-k, device feedback for the sampled token, and a passing
diagnostic logits reference. This closes the old full-prefix finite-logits
blocker, but not full-serving correctness. The current comparison is still
diagnostic rather than model-equivalent because prompt prefill was not
executed before the decode-position 17 readout; it records PTO top token
`220`, while the Hugging Face reference top token is `151667`.

The generated-token feedback path now uses one prompt-ring contract across
host feedback, device logits feedback, and embedding lookup. Earlier
policy-length diagnostic rows that predate the feedback-ring fix prove
scheduler progress and logits execution, but should not be treated as evidence
that long decode steps consumed the previous sampled token.

The bounded prompt-prefill readout path now also avoids a diagnostic RMSNorm
scale bug where attaching decode-position metadata made `scalar_args[1]`
present but zero, causing fallback RMSNorm paths to zero the hidden state.
Offset-aware activation sampling now reports the actual readout activation
buffer instead of a stale local-index buffer. The eight-layer MPK bounded
prefill path reaches nonzero readout logits and a second full selected DAG
decode after device feedback, but it remains diagnostic because it uses
bounded active projection/logit columns and only eight layers.
The same corrected path now scales to all 36 layers with bounded projection
and logits windows, including prompt prefill and a second full selected DAG
decode after device feedback. Full-vocabulary readout also writes finite,
nonzero logits, but it currently fails the checked diagnostic logits reference,
so full Qwen correctness remains open.

## Current Evidence

Structured paper-readiness evidence is tracked in
`evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json` and
`evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json`.
The review guards are under `.agents/checks/`, including the changelog,
benchmark-viewer data, and NVIDIA review-readiness checks.
The VDCores shared-instruction window plan is now guarded by
`.agents/skills/cuda-backend-eval/scripts/vdcores_validate_instruction_window_plan.py`;
the current audit records `window_contract_validation=pass` while preserving
`runnable_handoff_contract_status=required_not_implemented`, so the VDCores
paper row remains non-importable until runtime/builder support exists.

Recent raw A100 evidence stays under `tmp/`:

- `tmp/cuda-backend/repro-no-single-context/qwen-runner.json` records the
  pre-fix handoff failure for generated PTO full-serving commands:
  resource-backed execution returned
  `reason=single_context_live_session_required`.
- `tmp/cuda-backend/repro-single-context-prefill/qwen-runner.json` records the
  post-fix one-layer MPK handoff smoke with `prompt_prefill_executed`, 18
  executed prompt positions, zero prompt-prefill scheduler errors, and a
  passing readout-only first decode packet. This proves the generated command
  shape reaches live CUDA execution, but it is still diagnostic rather than
  model-equivalent full-serving correctness evidence.
- `tmp/cuda-backend/qwen-prefill-layer8-mpk-2026-06-03-rmsnorm-scale-fix/qwen-runner.json`
  records the bounded eight-layer MPK prompt-prefill readout after the
  diagnostic RMSNorm scale fix. It reports `prompt_prefill_executed`, 18
  prompt positions, zero scheduler errors, offset-aware readout
  `final_norm.max_abs_finite=3.182983`,
  `logits_summary.nonzero_count=32768`,
  `diagnostic_reference.status=pass`, and device feedback for sampled token
  `341`.
- `tmp/cuda-backend/qwen-prefill-layer8-mpk-2step-2026-06-03-rmsnorm-scale-fix/qwen-runner.json`
  records the follow-up bounded two-step MPK run. The first decode step reuses
  the prefilled hidden state for readout and samples token `341`; the second
  step runs the full eight-layer selected DAG at decode position 18, completes
  59/59 tasks with zero scheduler errors, reports nonzero bounded logits,
  passes the diagnostic logits reference, and samples token `82`.
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-2step-2026-06-04-rmsnorm-scale-fix/qwen-runner.json`
  records the corrected full 36-layer bounded MPK prompt-prefill path. It
  reports `prompt_prefill_executed`, 18 prompt positions, 4,554 prefill task
  completions, zero prefill scheduler errors, two decode steps, a second full
  selected DAG at decode position 18 with 255/255 completed tasks and zero
  scheduler errors, nonzero bounded logits, diagnostic logits reference pass,
  and device feedback tokens `[1647, 839]`.
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-status-gated/qwen-runner.json`
  records the full-vocabulary readout attempt after the artifact status gate
  was fixed. It reports full logits-buffer coverage, all finite/nonzero logits,
  top token `71590`, and clean scheduler counters, but the checked diagnostic
  logits reference fails with `mismatch_count=80` and
  `max_abs_error=2.793e-05`; the artifact now correctly reports
  `resource_backed_execution.status=fail`.
- `tmp/cuda-backend/qwen-full-model-reference-mpk-1step-2026-06-03/`
  records the current Hugging Face comparison failure.
- `tmp/cuda-backend/qwen-attention-dot-product-first-layer-2026-06-03/`
  records the dot-product attention smoke with zero scheduler errors.
- `tmp/cuda-backend/qwen-rotate-half-rope-full-descriptor-1step-mpk-2026-06-03/`
  records the rotate-half RoPE full-descriptor run with zero scheduler errors.
- `tmp/cuda-backend/qwen-projection-full-first-layer-2026-06-03-default-cache/`
  records first-layer full QKV and MLP projection-column execution with zero
  scheduler errors.
- `tmp/cuda-backend/qwen-attention-o-projection-window-256-first-layer-2026-06-03/`
  records bounded attention-output projection-window execution with zero
  scheduler errors.
- `tmp/cuda-backend/qwen-attention-o-cached-full-projection-first-layer-2026-06-03/`
  records cached attention-output execution with full first-layer projection
  windows and zero scheduler errors.
- `tmp/cuda-backend/qwen-prefill-two-step-first-layer-2026-06-03/`
  records prompt-prefill to readout-only to full-DAG decode feedback.
- `tmp/cuda-backend/qwen-feedback-ring-bounded-logits-mpk-2026-06-03/`
  records a 49-step first-layer MPK-policy diagnostic where device feedback
  observes wrapped input slots at decode positions 63 and 64, and the bounded
  diagnostic logits reference passes over 2,048 active-window logits.
- `tmp/cuda-backend/qwen-attention-batch-rows-active-prompt-mpk-2026-06-03/`
  records a full-prefix, full-logits MPK-policy run with 255 completed tasks
  and zero scheduler errors after the attention batch-row shape fix. It still
  reports empty row-0 top-k and a diagnostic logits reference failure, so it is
  not full-Qwen correctness evidence.
- `tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/`
  records the activation-row localization run. The pre-fix one-layer artifact
  found row-0 NaNs first at `layer_0_mlp_down` column 2048; the post-fix
  artifact `qwen-layer1-after-mlp-residual-fix.json` has no row-0 non-finite
  activations, full finite logits, row-0 top-k, and a passing diagnostic
  logits reference.
  The same directory now includes `qwen-layer2-after-mlp-residual-fix.json`,
  `qwen-layer4-after-mlp-residual-fix.json`, and
  `qwen-layer8-after-mlp-residual-fix.json`; the eight-layer artifact reports
  59/59 completed tasks, zero scheduler errors, no row-0 non-finite
  activations, full finite logits, populated top-k, and
  `diagnostic_reference.status=pass`.
  It also includes
  `qwen-full-prefix-after-mlp-residual-fix-420s.json`, where the full
  36-layer prefix reports 255/255 completed tasks, zero scheduler errors, no
  row-0 non-finite activations, full finite logits, populated top-k,
  `diagnostic_reference.status=pass`, and device feedback for sampled token
  `220`. The generated comparison record
  `qwen-full-prefix-hf-token-comparison.json` still fails the Hugging Face
  agreement gate with
  `comparison_scope=diagnostic_decode_without_prompt_prefill` and
  `blocking_reasons=[prompt_prefill_not_executed, token_mismatch]`: PTO
  selects token `220` while the Hugging Face reference selects token
  `151667`.
- `tmp/cuda-backend/qwen-prefill-readout-full-projection-mpk-2026-06-03/`
  records no JSON artifact; the local A100 full 36-layer prompt-prefill
  attempt with full projection and logits columns was stopped after saturating
  GPU 0 at about 27 GiB.

## Promotion Gate

Close this gap only after PTO rows for both `mpk_offline_decode` and
`vdcores_offline_decode` satisfy the full-serving import gate:

- `serving_coverage=full_serving`;
- `correctness_scope=full_qwen_numerical_correctness`;
- token/logit agreement against the Hugging Face Qwen/Qwen3-8B reference;
- `model_equivalent_ready=true` with
  `comparison_scope=model_equivalent_decode`;
- positive latency and throughput metrics for policy-length runs.

## Next Actions

- Close model-equivalent prompt-prefill and Hugging Face token/logit
  agreement. Prioritize model-correct prefill, KV-cache state, attention, and
  decode semantics over additional diagnostic-only scalar task-body evidence.
- Re-run the Hugging Face comparison after each kernel-fidelity fix.
- Capture policy-length MPK and VDCores serving rows only after correctness
  passes.
