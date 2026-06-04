# 2026-06-04 Qwen Layer 2-4 Model-Equivalent Boundary

## Code And Data Changed

- Captured bounded full-QKV model-equivalent MPK probes for two, three, and
  four Qwen decoder layers.
- Compared each probe against the local Hugging Face Qwen/Qwen3-8B reference
  over the first 512 logits at the prompt readout position.
- Updated the Qwen full-serving correctness gap with the narrowed divergence
  boundary.

## Architecture Quality

This slice moves the work back to benchmark-model execution evidence. It does
not add a new test contract or viewer field. The artifacts show that the
model-equivalent path is correct through two full-QKV layers over the bounded
first-512 readout window, then begins drifting in deeper layers.

The evidence narrows the next debugging target: layer 0/1 handoff, full-QKV
coverage, full RMSNorm scale, and logits projection are no longer the current
blocker for the bounded first-512 path. The next useful work is to inspect the
layer-2 to layer-3 transition, especially attention or MLP numeric drift that
can reorder close logits at the top-k boundary.

## Evaluation Run

PTO layer-prefix probes:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count {2,3,4} \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent
```

Hugging Face probes used the stored prompt token IDs from the existing
layer-36 hidden probe under
`tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-float32-reference/`
and loaded the local reference offline from
`tmp/sources/qwen3-8b-local-hf-reference`.

Results:

- Layer 2 comparison:
  `tmp/cuda-backend/qwen-prefill-layer2-mpk-1step-2026-06-04-model-equivalent-samples/`
  reports `status=pass` and matching first-512 top tokens
  `[10, 200, 58, 219, 368]`.
- Layer 3 comparison:
  `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-samples/`
  reports `status=partial_match`, `matching_topk_prefix=4`, PTO tokens
  `[10, 167, 376, 475, 58]`, and HF tokens
  `[10, 167, 376, 475, 229]`.
- Layer 4 comparison:
  `tmp/cuda-backend/qwen-prefill-layer4-mpk-1step-2026-06-04-model-equivalent-samples/`
  reports `status=partial_match`, `matching_topk_prefix=3`, PTO tokens
  `[411, 10, 368, 483, 473]`, and HF tokens
  `[411, 10, 368, 167, 473]`.

## Remaining Gaps

Full Qwen correctness remains open. The next targeted slice should compare
per-task layer-2 and layer-3 activation samples against Hugging Face internals
to identify whether the first drift comes from attention, post-attention
normalization, or MLP output.
