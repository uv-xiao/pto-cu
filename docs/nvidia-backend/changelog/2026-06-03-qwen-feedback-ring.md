# 2026-06-03 Qwen Feedback Ring

## Code And Data Changed

- Updated generated `qwen_embedding_lookup` source to read decode input IDs
  from `decode_position % prompt_stride`.
- Updated generated `qwen_logits` device feedback source to write sampled
  tokens to `(decode_position + 1) % prompt_stride`.
- Added source-contract regression coverage so the host feedback ring policy
  and device task bodies stay aligned.

## Architecture Quality

Policy-length decode now has one consistent input-token ring contract across
host feedback, device logits feedback, and embedding lookup. Before this
change, host-side metadata wrapped positions beyond the runtime prompt stride,
but the device logits body skipped those writes while embedding clamped reads
to the last prompt slot. Long decode diagnostics could therefore report
completed scheduler work without proving generated-token chaining.

## Evaluation Run

Focused regression first failed on the missing generated-source wrap, then
passed after the task-body update:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_decode_feedback.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  -q -k 'feedback_wraps_prompt_ring_source or falls_back_when_position_exceeds_prompt_stride'
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
