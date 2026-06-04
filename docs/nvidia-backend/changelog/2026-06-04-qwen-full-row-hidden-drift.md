# 2026-06-04 Qwen Full-Row Hidden Drift

## Code And Data Changed

- Added an opt-in
  `--resource-backed-activation-row-dump-descriptor-ids` diagnostic flag for
  resource-backed Qwen runs.
- Kept default activation summaries compact; `row_values` is emitted only for
  descriptor IDs named by the caller.
- Captured a layer-3 MPK model-equivalent probe that dumps the full
  4,096-column `final_norm` row.

## Architecture Quality

This slice is runtime debugging evidence, not another viewer contract. The
diagnostic exists to answer one model-correctness question: whether the layer-3
rank-5 swap comes from a sparse bad column or from accumulated row-wide numeric
drift.

The answer is now concrete. The hidden row is close in aggregate, but the
distributed error is enough to flip the token `229` versus token `58`
boundary. That points the next fix toward full-row reductions or accumulation
order in the layer-2 to layer-3 path, not descriptor plumbing or logits
projection.

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
  --resource-backed-activation-row-dump-descriptor-ids final_norm
```

Artifacts:

- Artifact root:
  `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-full-row/`
- `qwen-runner.json`
- `pto-hf-layer3-full-row-comparison.json`

Results:

- The PTO runner reports `resource_backed_execution.status=pass`.
- The dumped `final_norm.row_values` has 4,096 columns and
  `column_sample_policy=full_row`.
- PTO first-512 top-k remains `[10, 167, 376, 475, 58]`.
- Hugging Face first-512 top-k remains `[10, 167, 376, 475, 229]`.
- Full-row hidden drift is small but distributed:
  `mean_abs_hidden_delta=0.007585`, `p50=0.005848`, `p99=0.028655`, and
  `max_abs_hidden_delta=0.331558`.
- The hidden-delta contribution to the token-229-minus-token-58 boundary is
  `-0.041158`, moving the boundary from Hugging Face `0.031095` to PTO
  `-0.010063`.

## Remaining Gaps

Full Qwen correctness remains open. The next benchmark-model slice should
compare full-row reductions or accumulation order inside the layer-2 to
layer-3 transition, especially attention-output and MLP reduction paths.
