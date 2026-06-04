# 2026-06-04 Qwen Layer-1 Model-Equivalent Samples

## Code And Data Changed

- Added a compact `value_sample` field to activation finiteness summaries.
- Kept the sample bounded to the first four row values, with JSON-safe
  sentinels for non-finite values.
- Captured a one-layer model-equivalent Qwen artifact with per-task activation
  samples and compared its first-512 logits against Hugging Face.

## Architecture Quality

The activation summary still stays small, but it now exposes enough data to
compare PTO and Hugging Face at specific task boundaries. This avoids
guessing from max-abs magnitudes alone and helps keep future debugging focused
on the first divergent layer or projection window.

## Evaluation Run

Focused TDD regression:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_activation_finiteness_summary_reports_row_local_nonfinite_column \
  -q
```

Result: failed before the `value_sample` field existed, then passed after the
summary update.

One-layer live A100 evidence:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 1 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --output-json tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-model-equivalent-samples/qwen-runner.json
```

Result: the artifact passed. The final-norm sample begins
`[0.232644, -0.81544, -0.02089, -0.415388]`, matching the Hugging Face
layer-1 final-norm sample `[0.232788, -0.814453, -0.020905, -0.415283]`.

The Hugging Face first-512 logits probe wrote
`tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-model-equivalent-samples/hf-layer1-first512-logits-probe.json`.
PTO and Hugging Face both selected top tokens `[200, 68, 475, 10, 58]` over
the first 512 vocab columns, with logits differing by about `0.003` or less.

Post-fix checks:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_activation_finiteness_summary_reports_row_local_nonfinite_column \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_resource_backed_logits_summary_reports_row0_token_ids \
  -q
```

Result: both focused tests passed.

## Remaining Gaps

The one-layer, first-512 model-equivalent window now agrees with Hugging Face.
Full Qwen correctness remains open for deeper decoder layers, full-vocabulary
ranking, and policy-length MPK/VDCores serving rows.
