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

The generated-token feedback path now uses one prompt-ring contract across
host feedback, device logits feedback, and embedding lookup. Earlier
policy-length diagnostic rows that predate the feedback-ring fix prove
scheduler progress and logits execution, but should not be treated as evidence
that long decode steps consumed the previous sampled token.

## Current Evidence

Structured paper-readiness evidence is tracked in
`evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json` and
`evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json`.
The review guards are under `.agents/checks/`, including the changelog,
benchmark-viewer data, and NVIDIA review-readiness checks.

Recent raw A100 evidence stays under `tmp/`:

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
- positive latency and throughput metrics for policy-length runs.

## Next Actions

- Continue replacing diagnostic scalar task-body math with model-correct Qwen
  kernels.
- Extend cached attention-output projection evidence from first-layer smoke to
  all selected layers and full logits when runtime is practical.
- Re-run the Hugging Face comparison after each kernel-fidelity fix.
- Capture policy-length MPK and VDCores serving rows only after correctness
  passes.
