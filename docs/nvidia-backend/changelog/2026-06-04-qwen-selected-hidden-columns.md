# 2026-06-04 Qwen Selected Hidden Columns

## Code And Data Changed

- Added an optional
  `--resource-backed-activation-sample-columns` diagnostic flag for
  resource-backed Qwen runs.
- Threaded selected activation columns into activation finiteness summaries so
  a run can report exact hidden dimensions without dumping full rows.
- Captured a layer-3 MPK model-equivalent probe for the eight hidden columns
  with the largest Hugging Face contribution to the token-229-minus-token-58
  logit boundary.

## Architecture Quality

This is a narrow benchmark-model diagnostic, not a new viewer field family or
artifact-specific test suite. It keeps the broad finiteness summary intact and
adds only selected values when a caller asks for them.

The evidence moves the investigation away from sparse stage plumbing. The
sampled high-impact hidden columns are close, so the remaining layer-3 rank-5
swap is more likely accumulated full-row numeric drift than one missing or
badly wired hidden dimension.

## Evaluation Run

PTO probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 3 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --resource-backed-activation-sample-columns 3666,676,2275,4081,2202,822,2316,2652
```

Artifacts:

- Artifact root:
  `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-selected-columns/`
- `hf-layer3-logit-diff-columns.json`
- `qwen-runner.json`
- `pto-hf-layer3-selected-column-comparison.json`

Results:

- `pto-hf-layer3-selected-column-comparison.json` reports `status=pass`,
  `selected_column_count=8`, `max_abs_hidden_delta=0.266905`, and
  `max_abs_contribution_delta=0.008797`.
- The summed selected-column contribution to token `229` minus token `58` is
  close: PTO `-0.821749`, Hugging Face `-0.798205`.
- The diagnostic did not perturb the prior layer-3 PTO first-512 top-k:
  `[10, 167, 376, 475, 58]`.

## Remaining Gaps

Full Qwen correctness remains open. The next useful benchmark-model slice
should compare full-row reductions or wider hidden/logit aggregates rather
than adding more sparse per-column tests.
