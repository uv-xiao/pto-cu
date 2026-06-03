# 2026-06-03 Qwen Feedback Ring

## Code And Data Changed

- Updated generated `qwen_embedding_lookup` source to read decode input IDs
  from `decode_position % prompt_stride`.
- Updated generated `qwen_logits` device feedback source to write sampled
  tokens to `(decode_position + 1) % prompt_stride`.
- Updated host-side `feedback_input_index()` to use the same modulo policy
  when observing device-committed feedback.
- Added source-contract regression coverage so the host feedback ring policy
  and device task bodies stay aligned.

## Architecture Quality

Policy-length decode now has one consistent input-token ring contract across
host feedback, device logits feedback, and embedding lookup. Before this
change, the device logits body skipped writes beyond the runtime prompt stride,
embedding clamped reads to the last prompt slot, and host observation collapsed
all wrapped positions to slot 0. Long decode diagnostics could therefore report
completed scheduler work without proving generated-token chaining.

## Evaluation Run

Focused regression first failed on the missing generated-source wrap, then
passed after the task-body update:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_decode_feedback.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  -q -k 'feedback_wraps_prompt_ring_source or wraps_when_position_exceeds_prompt_stride'
```

Result after the fix: `2 passed, 10 deselected`.

Adjacent Qwen task-body and decode-feedback regression suite passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_decode_feedback.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `12 passed`.

A100 first-layer MPK-policy feedback-ring smoke passed scheduler and device
feedback checks:

```bash
ARTIFACT=tmp/cuda-backend/qwen-feedback-ring-first-layer-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock \
  --workspace-cuda-live \
  --single-context-live-session \
  --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-decode-steps 49 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols 128 \
  --resource-backed-logits-active-cols 128 \
  --device 0 \
  --output-json "$ARTIFACT/qwen-decode-loop-runner.json"
```

Result: `tmp/cuda-backend/qwen-feedback-ring-first-layer-mpk-2026-06-03/`.
The artifact records 49 executed decode steps, 490 task completions, zero
scheduler errors, and `device_token_feedback_observed` for every step. The
ring boundary is visible: decode position 63 writes `next_input_index = 0`,
and decode position 64 writes `next_input_index = 1`. The final diagnostic
logits reference fails after the long bounded loop, so this artifact is
token-feedback-ring evidence only, not logits-correctness evidence.

An A100 full 36-layer prompt-prefill/readout attempt with full projection and
full logits columns was started under
`tmp/cuda-backend/qwen-prefill-readout-full-projection-mpk-2026-06-03/`.
It was stopped after saturating GPU 0 at about 27 GiB without producing a JSON
artifact. This is not correctness evidence; it confirms the current scalar
full-projection prefill path is still impractical locally.

## Remaining Gaps

This fixes generated-token chaining for bounded/policy-length diagnostics, but
does not close full Qwen numerical correctness. Promotion still requires
token/logit agreement against the Hugging Face Qwen/Qwen3-8B reference and
policy-length MPK plus VDCores rows with positive latency/throughput metrics.
