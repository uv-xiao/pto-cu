# 2026-06-04 Qwen Layer 2-3 Stage Samples

## Code And Data Changed

- Captured Hugging Face intermediate samples for layer-2 and layer-3 Qwen
  stages at the same prompt readout position used by the PTO layer-prefix
  probes.
- Compared PTO and Hugging Face samples for input norm, raw QKV,
  RoPE-applied Q/K, attention output, post-attention norm, MLP activation
  product, and layer output.
- Recorded selected Hugging Face logits for the layer-3 top-k boundary.

## Architecture Quality

This keeps the investigation on model execution evidence rather than adding
another test or viewer contract. The sampled stages are close through layer 3,
including RoPE-applied Q/K, so the current evidence does not support a missing
task, wrong descriptor handoff, or skipped RMSNorm/attention stage.

The remaining blocker is now sharper: the first visible mismatch is a small
ranking drift at the layer-3 first-512 top-k boundary. That points toward
accumulated numeric differences across unsampled columns or full-row
reductions, not an obvious sampled-column stage break.

## Evaluation Run

Artifacts:

- Artifact root:
  `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-samples/`
- `hf-layer2-3-stage-samples.json`
- `hf-layer2-3-rope-qk-samples.json`
- `pto-hf-layer2-3-stage-comparison.json`
- `hf-layer3-selected-logits.json`

Results:

- JSON validation for the generated comparison artifacts passed with
  `jq empty`.
- `pto-hf-layer2-3-stage-comparison.json` reports `status=pass` and
  `largest_sample_error=0.018348` across comparable sampled stages.
- RoPE-applied Q/K samples match the PTO `qwen_attention_qk_norm` buffer:
  layer 2 PTO starts `[-0.080086, -1.094227, -0.92434, 0.867864]` and HF
  starts `[-0.082031, -1.085938, -0.925781, 0.863281]`.
- The layer-3 selected-logit probe shows the first top-k boundary swap is
  close but real in the HF reference: token `229` has logit `5.5`, while PTO's
  rank-5 token `58` has HF logit `5.46875`.

## Remaining Gaps

Full Qwen correctness remains open. The next useful diagnostic should compare
more than the first four sampled columns, ideally the selected logit-driving
hidden dimensions or full-row reductions across the layer-2 to layer-3
transition.
