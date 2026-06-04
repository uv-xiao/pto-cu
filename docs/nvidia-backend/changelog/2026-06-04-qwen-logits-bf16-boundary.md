# 2026-06-04 Qwen Logits BF16 Boundary

## Code And Data Changed

- Rounded generated Qwen logits output writes to the Hugging Face bf16 tensor
  boundary.
- Matched the host diagnostic logits reference to the same bf16 output
  boundary.
- Relaxed the plan-history currentness guard only when the current commit
  updates `plan_history.json`, so runtime commits can archive their parent
  without allowing stale archives for ordinary runtime-only commits.
- Added focused contracts for the generated logits write, host diagnostic
  reference, and plan-history parent exception.

## Architecture Quality

This slice stays on benchmark-model correctness. Hugging Face exposes logits
as bf16 tensor values for the layer-prefix replay, while the generated CUDA
task body was storing the raw FP32 accumulator into the logits buffer. The
runtime now makes that output boundary explicit, and the host diagnostic
reference checks the same contract.

The tests remain narrow. They check the source-level boundary and one scalar
host-reference value instead of adding another large sparse artifact matrix.

## Evaluation Run

Focused red/green checks:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_qwen_logits_match_hf_bf16_output_boundary \
  -q
```

Result: failed before the generated CUDA write rounded `acc`; passed after the
runtime change (`1 passed`).

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_projection_matches_bf16_output_boundary \
  -q
```

Result: failed before the host reference rounded the accumulator; passed after
the reference change (`1 passed`).

Runtime probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  timeout 420s .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 3 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --resource-backed-activation-row-dump-descriptor-ids \
  layer_2_input_norm,layer_2_attention_qkv,layer_2_attention_qk_norm
```

Artifact:
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-logits-bf16-boundary/`.

Result: the runner passed with zero scheduler errors and diagnostic logits
reference `status=pass`, `max_abs_error=0.0`. First-512 PTO top-k changed from
`[(10, 6.839146), (167, 6.003837), (376, 5.650578), (475, 5.533737),
(58, 5.498676)]` to `[(10, 6.84375), (167, 6.0), (376, 5.65625),
(475, 5.53125), (58, 5.5)]`.

## Remaining Gaps

Full Qwen correctness remains open. Hugging Face first-512 top-k is still
`[(10, 6.84375), (167, 6.03125), (376, 5.59375), (475, 5.53125),
(229, 5.5)]`; PTO still ranks token `58` fifth after bf16 rounding. The next
slice should stay on upstream hidden/ranking drift rather than adding another
viewer or log-format guard.
