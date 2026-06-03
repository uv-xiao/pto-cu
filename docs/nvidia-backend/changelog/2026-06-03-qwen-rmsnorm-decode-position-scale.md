# 2026-06-03 Qwen RMSNorm Decode-Position Scale

## Code And Data Changed

- Fixed diagnostic Qwen RMSNorm fallback paths so attaching decode-position
  metadata in `scalar_args[2]` does not turn a zero `scalar_args[1]` slot into
  an external scale of zero.
- Made resource-backed activation-finiteness sampling honor readout packet
  activation-buffer offsets.
- Added focused regressions for the offset-aware sampler and the neutral
  diagnostic RMSNorm scale source.

## Architecture Quality

The prompt-prefill readout path now reports the actual readout activation
buffer and no longer zeroes `final_norm` just because decode-step state is
attached. This keeps diagnostic readout evidence aligned with the live
activation chain instead of sampling stale buffer indices.

## Evaluation Run

The bounded eight-layer MPK prompt-prefill run after the fix wrote
`tmp/cuda-backend/qwen-prefill-layer8-mpk-2026-06-03-rmsnorm-scale-fix/qwen-runner.json`.
Result: `prompt_prefill.status=prompt_prefill_executed`, 18 prompt positions,
zero scheduler errors, readout `final_norm.max_abs_finite=3.182983`,
`logits_summary.nonzero_count=32768`, diagnostic logits reference pass with
`max_abs_error=8e-07`, and device feedback sampled token `341`.

The two-step bounded MPK run wrote
`tmp/cuda-backend/qwen-prefill-layer8-mpk-2step-2026-06-03-rmsnorm-scale-fix/qwen-runner.json`.
Result: two decode steps executed, the second full selected DAG completed
59/59 tasks with zero scheduler errors, `logits_summary.nonzero_count=32768`,
diagnostic logits reference pass with `max_abs_error=4.9e-07`, and device
feedback sampled tokens `[341, 82]`.

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_rmsnorm_diagnostic_fallback_keeps_decode_position_scale_neutral \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_readout_activation_sampling_uses_packet_offset \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  -q
```

Result: three tests passed.

## Remaining Gaps

This is still bounded diagnostic MPK evidence. PTO still needs full-vocab,
full-layer, model-equivalent Qwen token/logit agreement against Hugging Face
and full-serving MPK/VDCores rows with latency and throughput before the paper
claim can close.
