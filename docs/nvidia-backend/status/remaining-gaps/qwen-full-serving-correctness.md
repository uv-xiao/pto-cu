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
nonzero logits and passes the checked diagnostic logits reference after the
host-side diagnostic reference was changed to accumulate like the generated
float32 CUDA logits kernel. Full Qwen correctness remains open because the
current full-vocabulary pass is still diagnostic rather than Hugging Face
model-equivalent token/logit agreement.
The latest model-equivalent comparison reaches prompt-prefill execution and
full-vocabulary diagnostic agreement, but still selects PTO token `71590`
while Hugging Face selects `151667`. A hidden-state probe shows the mismatch is
upstream of logits: PTO final norm begins `[-4.339333, 0.982281, -10.556374,
4.683925]`, while the Hugging Face final hidden begins `[-0.181641,
-0.644531, 0.6875, -2.0625]`.
The next localized blocker was attention-output execution. Diagnostic-mode
`qwen_attention_o` launch packets were dropping
`attention_o_projection_input_count`, which made the kernel project zero
columns. That scalar binding is now fixed, but bounded QKV projection windows
can still prune K/V writes because K starts after the 4,096-column Q region.
Model-equivalent Qwen checks therefore need full QKV projection coverage, not
only full logits.
They also need the model-equivalent numeric task mode. A one-layer full-QKV
run with the old default diagnostic mode reached final-norm
`max_abs_finite=3.213728`, while the same one-layer full-QKV run with full
RMSNorm reached `max_abs_finite=43.317745`, matching the Hugging Face
layer-1 final-norm scale `43.34375`. The serving command plan now passes
`--resource-backed-numeric-task-mode model_equivalent` so paper-target PTO
runs do not silently use the external-scale diagnostic RMSNorm path.
With activation value samples enabled, the one-layer model-equivalent run now
matches the Hugging Face layer-1 sampled final norm and first-512 logits
top-k: PTO and HF both select tokens `[200, 68, 475, 10, 58]` over that bounded
vocab window. The remaining model-equivalence gap is therefore beyond the
sampled one-layer/first-512 window: deeper layers, full-vocabulary ranking, or
policy-length decode still need proof.
The deeper full-QKV prefix probes narrow that boundary further. The two-layer
model-equivalent MPK probe matches the Hugging Face first-512 top-5 tokens
exactly: `[10, 200, 58, 219, 368]`. The three-layer probe matches the top four
tokens and differs at the fifth rank, while the four-layer probe matches the
top three tokens and differs at rank four. The remaining gap is therefore a
small but accumulating deeper-layer numeric/ranking divergence, not a
layer-0/1 handoff, RMSNorm-scale, or logits-projection failure.
The layer-2/3 stage-sample comparison does not reveal a wrong substage in the
sampled columns. PTO and Hugging Face match closely at layer-2 input norm, raw
QKV, RoPE-applied Q/K, attention output, post-attention norm, MLP activation
product, and layer output; the largest sampled value error across comparable
layer-2/3 stages is `0.018348`. The first visible mismatch remains a
close-logit ranking drift: after layer 3, Hugging Face scores token `229` at
`5.5` and PTO's rank-5 token `58` at `5.46875`.
The selected-column follow-up samples the eight hidden dimensions with the
largest Hugging Face contribution to the token-229-minus-token-58 logit
difference. PTO and Hugging Face remain close on those dimensions:
`max_abs_hidden_delta=0.266905` and `max_abs_contribution_delta=0.008797`.
That makes a single bad high-contribution hidden column unlikely; the next
target is full-row accumulated numeric drift, especially reductions that are
not covered by sparse samples.
The full-row layer-3 follow-up confirms that direction. The dumped
4,096-column PTO final-norm row has small distributed drift versus Hugging
Face (`mean_abs_hidden_delta=0.007585`, `p99=0.028655`,
`max_abs_hidden_delta=0.331558`), but the aggregate effect on the close
token-229-minus-token-58 boundary is `-0.041158`: PTO scores the boundary at
`-0.010063`, while Hugging Face scores it at `0.031095`. The blocker is now a
row-wide accumulated numeric drift, not a missing final-norm row, logits
projection failure, or one isolated high-impact hidden column.
The full-row layer-2 stage comparison also rules out a gross last-layer MLP
handoff failure. After fixing the activation-summary diagnostic to include a
trailing non-logits task, the layer-2 `mlp_down` row is available and remains
close to Hugging Face (`mean_abs_delta=0.000989`, `p99=0.003402`,
`max_abs_delta=0.081783`). The largest row-wide substage delta is
RoPE-applied Q/K (`mean_abs_delta=0.005142`, `p99=0.024234`), while
attention-O remains close (`mean_abs_delta=0.000497`, `p99=0.001711`).
The next root-cause target is therefore precision/rounding accumulation across
QK/RoPE and final normalization, not descriptor ordering or an omitted MLP
output.
The logits boundary follow-up now also writes generated Qwen logits at the
same bf16 output boundary used by the Hugging Face replay. This removes the
host/device diagnostic reference mismatch for the fresh layer-3 full-stage
probe, but it does not close model-equivalent top-k agreement: PTO still
returns `[10, 167, 376, 475, 58]`, while Hugging Face returns
`[10, 167, 376, 475, 229]`. After bf16 rounding, PTO token `58` and Hugging
Face token `229` both sit at `5.5`, so the remaining blocker is a close
ranking/tie boundary or upstream hidden drift rather than a raw logits-output
dtype mismatch.
The MLP activation follow-up then rounds `silu(gate_proj) * up_proj` to the
bf16 tensor boundary before `down_proj`. That moves PTO token `58` from `5.5`
to the Hugging Face value `5.46875`, while preserving a passing diagnostic
logits reference. Full layer-3 top-k still remains open because PTO does not
yet surface Hugging Face token `229` in the first-512 top five.
The attention-context follow-up rounds the softmax-weighted value vector to
bf16 before `o_proj`. That improves common top-logit agreement for tokens
`10`, `167`, and `376`, while token `58` moves back to `5.5` and token `229`
still remains missing from PTO's first-512 top five.

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
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-mismatch-diagnostics/qwen-runner.json`
  records the same full-vocabulary path with mismatch-index diagnostics before
  the reference-precision fix. It localizes the first checked mismatch to
  logits index `80` and the max-error mismatch to logits index `234`, both at
  ordinary finite logit magnitudes, which points to host reference precision
  rather than scheduler, buffer, or tile-index failure.
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-float32-reference/qwen-runner.json`
  records the post-fix full-vocabulary 36-layer MPK prompt-prefill readout. It
  reports `resource_backed_execution.status=pass`, one passing workload, full
  logits-buffer coverage, 2,430,976 finite/nonzero logits, top token `71590`,
  and checked diagnostic reference `status=pass`, `mismatch_count=0`,
  `max_abs_error=0` over 3,904 checked full-vocabulary elements.
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-float32-reference/hf-comparison.json`
  records the current model-equivalent Hugging Face comparison failure:
  `model_equivalent_ready=true`, `comparison_scope=model_equivalent_decode`,
  `blocking_reasons=[token_mismatch]`, PTO token `71590`, and Hugging Face
  token `151667`.
- `tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-float32-reference/hf-hidden-probe.json`
  records the local Hugging Face hidden-state probe that shows the mismatch is
  before the logits projection.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-attention-o-scalar/qwen-runner.json`
  records the one-layer bounded projection rerun after the attention-O scalar
  binding fix. It still reports zero attention-O output because
  `--resource-backed-projection-active-cols 512` computes only Q columns and
  prunes K/V cache writes.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-attention-o/qwen-runner.json`
  records the one-layer full-QKV rerun after the scalar fix. It reports
  nonzero `layer_0_attention_qkv.max_abs_finite=0.02274`, nonzero
  `layer_0_attention_o.max_abs_finite=0.062311`, and a passing diagnostic
  logits reference.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-attention-o/hf-layer1-norm-probe.json`
  compares the old default-mode one-layer PTO final-norm sample with Hugging
  Face `model.norm(hidden_states[1])`, showing that the layer-0 path already
  diverges before later decoder layers.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-full-rmsnorm/qwen-runner.json`
  records the same one-layer full-QKV path with full RMSNorm enabled. It
  reports `layer_0_input_norm.max_abs_finite=0.286054`,
  `layer_0_attention_o.max_abs_finite=2.704043`, final-norm
  `max_abs_finite=43.317745`, and a passing diagnostic logits reference. This
  fixes the RMSNorm scale mode for model-equivalent attempts, but token/logit
  agreement still remains open.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-model-equivalent-samples/qwen-runner.json`
  records the one-layer model-equivalent rerun with per-task activation value
  samples. The final-norm sample begins `[0.232644, -0.81544, -0.02089,
  -0.415388]`, matching the Hugging Face layer-1 final-norm sample within the
  expected dtype/rounding range.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-model-equivalent-samples/hf-layer1-first512-logits-probe.json`
  compares PTO and Hugging Face over the first 512 logits after one layer.
  Both select top tokens `[200, 68, 475, 10, 58]`, with logits within about
  `0.003` absolute error.
- `tmp/cuda-backend/qwen-prefill-layer2-mpk-1step-2026-06-04-model-equivalent-samples/`
  records the two-layer full-QKV model-equivalent MPK probe and Hugging Face
  comparison. The generated comparison artifact reports
  `status=pass`, matching first-512 top tokens `[10, 200, 58, 219, 368]`, and
  final-norm sample absolute errors `[0.00445, 0.002582, 0.010898,
  0.000518]`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-samples/`
  records the three-layer full-QKV model-equivalent MPK probe. The comparison
  reports `status=partial_match`, `matching_topk_prefix=4`, PTO top tokens
  `[10, 167, 376, 475, 58]`, and Hugging Face top tokens
  `[10, 167, 376, 475, 229]`.
- `tmp/cuda-backend/qwen-prefill-layer4-mpk-1step-2026-06-04-model-equivalent-samples/`
  records the four-layer full-QKV model-equivalent MPK probe. The comparison
  reports `status=partial_match`, `matching_topk_prefix=3`, PTO top tokens
  `[411, 10, 368, 483, 473]`, and Hugging Face top tokens
  `[411, 10, 368, 167, 473]`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-samples/`
  also contains `hf-layer2-3-stage-samples.json`,
  `hf-layer2-3-rope-qk-samples.json`,
  `pto-hf-layer2-3-stage-comparison.json`, and
  `hf-layer3-selected-logits.json`. These record matching Hugging Face
  intermediate samples for layer-2 and layer-3 stages at prompt position 17.
  The stage comparison reports `largest_sample_error=0.018348` across
  comparable stage samples, including RoPE-applied Q/K samples. The selected
  logit probe records the layer-3 top-k boundary: token `229` is `5.5`, while
  token `58` is `5.46875`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-selected-columns/`
  records the selected-column layer-3 follow-up. The runner artifact includes
  `final_norm.selected_columns` for the eight HF top-contribution hidden
  columns, and `pto-hf-layer3-selected-column-comparison.json` reports
  `status=pass`, `max_abs_hidden_delta=0.266905`,
  `max_abs_contribution_delta=0.008797`, and matching prior PTO first-512
  top-k `[10, 167, 376, 475, 58]`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-full-row/`
  records the opt-in full-row layer-3 final-norm dump and Hugging Face
  comparison. The PTO runner reports `final_norm.row_values` with 4,096
  columns and unchanged first-512 top-k `[10, 167, 376, 475, 58]`.
  `pto-hf-layer3-full-row-comparison.json` reports `status=pass`,
  `mean_abs_hidden_delta=0.007585`, `p99=0.028655`,
  `max_abs_hidden_delta=0.331558`, and
  `token_229_minus_58.hidden_delta_contribution_sum=-0.041158`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-stage-full-rows-v2/`
  records the layer-2 full-row stage comparison after activation summaries
  started including trailing non-logits tasks. The runner artifact includes
  full rows for `layer_2_input_norm`, `layer_2_attention_qkv`,
  `layer_2_attention_qk_norm`, `layer_2_attention_o`,
  `layer_2_post_attention_norm`, `layer_2_mlp_gate_up`, and
  `layer_2_mlp_down`. The comparison artifact reports `status=pass`;
  `layer_2_mlp_down.mean_abs_delta=0.000989`, `p99=0.003402`; and
  `layer_2_attention_qk_norm.mean_abs_delta=0.005142`, `p99=0.024234`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-qk-bf16-boundary/`
  records the same layer-3 boundary after matching the Q/K RMSNorm and RoPE
  output boundary to Hugging Face bf16 tensor semantics. The first-512 top-k
  remains `[10, 167, 376, 475, 58]`, but distributed stage drift improves:
  `layer_2_attention_qk_norm.mean_abs_delta=0.004693`,
  `layer_2_attention_o.mean_abs_delta=0.000414`,
  `layer_2_post_attention_norm.mean_abs_delta=0.001792`, and
  `layer_2_mlp_down.mean_abs_delta=0.000847`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-qkv-bf16-boundary/`
  records the next runtime slice after rounding Q, K, and V projection outputs
  to the Hugging Face bf16 tensor boundary before Q/K RMSNorm and KV-cache
  writeback. The layer-0 comparison reports
  `layer_0_attention_qkv.mean_abs_delta=0.000030` and
  `layer_0_attention_qk_norm.mean_abs_delta=0.001969`. The layer-3 first-512
  top-k remains `[10, 167, 376, 475, 58]` while the Hugging Face replay remains
  `[10, 167, 376, 475, 229]`; layer-2 stage means are
  `layer_2_attention_qkv.mean_abs_delta=0.000218`,
  `layer_2_attention_qk_norm.mean_abs_delta=0.004728`,
  `layer_2_attention_o.mean_abs_delta=0.000453`,
  `layer_2_post_attention_norm.mean_abs_delta=0.001830`, and
  `layer_2_mlp_down.mean_abs_delta=0.000886`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-rmsnorm-bf16-boundary-slim/`
  records the next runtime slice after matching Qwen input, post-attention,
  and final RMSNorm outputs to Hugging Face bf16 tensor boundaries. The
  layer-0 full-row artifact at
  `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-rmsnorm-bf16-boundary/`
  improves `layer_0_attention_qkv.mean_abs_delta` to `0.000025` and
  `layer_0_attention_qk_norm.mean_abs_delta` to `0.001333`. The slim layer-3
  artifact improves the dumped layer-2 means to
  `layer_2_input_norm.mean_abs_delta=0.000082`,
  `layer_2_attention_qkv.mean_abs_delta=0.000204`, and
  `layer_2_attention_qk_norm.mean_abs_delta=0.004423`, but the first-512
  top-k remains `[10, 167, 376, 475, 58]` versus Hugging Face
  `[10, 167, 376, 475, 229]`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-residual-bf16-boundary-slim/`
  records the next runtime slice after matching attention-O output,
  post-attention residual input, and MLP residual output to Hugging Face bf16
  tensor boundaries. The layer-0 full-row artifact at
  `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-residual-bf16-boundary/`
  improves `layer_0_attention_o.mean_abs_delta` to `0.000089`,
  `layer_0_post_attention_norm.mean_abs_delta` to `0.000158`, and
  `layer_0_mlp_down.mean_abs_delta` to `0.000204`. The slim layer-3 artifact
  improves `layer_2_attention_qkv.mean_abs_delta` to `0.000194` and
  `layer_2_attention_qk_norm.mean_abs_delta` to `0.004318`, with QK/RoPE p99
  improving to `0.023438`; the first-512 top-k remains
  `[10, 167, 376, 475, 58]` versus Hugging Face
  `[10, 167, 376, 475, 229]`.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-logits-bf16-boundary/`
  records the next runtime slice after writing generated logits through the
  bf16 output boundary and matching the host diagnostic reference to that
  boundary. The runner reports `resource_backed_execution.status=pass`, zero
  scheduler errors, and diagnostic logits reference `status=pass` with
  `max_abs_error=0.0`. First-512 top-k remains
  `[10, 167, 376, 475, 58]`; the Hugging Face replay remains
  `[10, 167, 376, 475, 229]`, with PTO token `58` and Hugging Face token
  `229` tied at `5.5` after bf16 rounding.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-mlp-gate-bf16-boundary/`
  records the next runtime slice after rounding the MLP gate/up activation
  product to the bf16 tensor boundary before the down projection. The runner
  reports `resource_backed_execution.status=pass`, zero scheduler errors, and
  diagnostic logits reference `status=pass` with `max_abs_error=0.0`. First-512
  top-k remains `[10, 167, 376, 475, 58]`, but PTO token `58` moves from
  `5.5` to `5.46875`, matching the prior Hugging Face selected-logit probe
  for token `58`. Hugging Face top-k remains
  `[10, 167, 376, 475, 229]`, so the next blocker is token `229` or another
  upstream rank-boundary drift.
- `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-attention-context-bf16-boundary/`
  records the next runtime slice after rounding the softmax-weighted attention
  value vector to bf16 before output projection. The runner reports
  `resource_backed_execution.status=pass`, zero scheduler errors, and
  diagnostic logits reference `status=pass` with `max_abs_error=0.0`.
  Compared with the MLP-gate boundary artifact, common top-logit errors
  improve for token `10` (`6.8125` to `6.84375`, matching HF), token `167`
  (`5.96875` to `6.0`, HF `6.03125`), and token `376` (`5.65625` to
  `5.625`, HF `5.59375`). Token `58` regresses from its selected HF value
  `5.46875` back to `5.5`, and Hugging Face token `229` still does not appear
  in PTO's first-512 top five.
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

- Close Hugging Face model-equivalent token/logit agreement. Prioritize
  model-correct prefill, KV-cache state, attention, and decode semantics over
  additional diagnostic-only scalar task-body evidence.
- Re-run the Hugging Face comparison after each kernel-fidelity fix.
- Capture policy-length MPK and VDCores serving rows only after correctness
  passes.
