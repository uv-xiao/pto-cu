# 2026-06-02 Qwen Logits Tiled Vocab Projection

## Code And Data Changed

- Added descriptor-bounded hidden-by-vocab logits projection source in
  `examples/cuda/qwen_persistent_task_bodies_impl/logits_feedback.py`.
- Added the source contract symbols
  `qwen_logits_tiled_vocab_projection_source`,
  `qwen_logits_full_vocab_argmax_source`, and
  `qwen_logits_device_sampled_token_feedback_source` to the Qwen task-body
  manifest path.
- Added paper-matrix evidence pointing at the generated raw artifact under
  `tmp/cuda-backend/pto-serving-logits-tiled-vocab-projection-2026-06-02/`.

## Architecture Quality

The logits task body now uses explicit descriptor fields for the hidden width,
input stride, weight stride, and tile size. That keeps the projection source
reviewable as generated CUDA instead of hiding it behind a generic linear
helper, while preserving the existing persistent-device task-body ABI.

## Evaluation Run

Focused verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `6 passed in 0.13s`.

## Remaining Gaps

This is source-level hardening. Full Qwen serving correctness still needs a
resource-backed decode-loop run whose sampled-token feedback matches the model
reference, then paper-matrix import through the full-serving correctness gate.
