# Qwen Sampled Logits Reference

## Code And Data Changed

- Added `resource_logits_reference.py` for host-side Qwen logits summaries,
  BF16 tensor-argument decoding, bounded reference-index selection, and
  branch-specific diagnostic formulas.
- Kept `resource_graph.py` focused on device graph-state ownership and device
  memory copies by moving pure logits-reference helpers out of that file.
- Replaced the prior policy-length diagnostic viewer rows with rows from
  `tmp/cuda-backend/pto-serving-decode-loop-policy-sampled-ref-5948323b/`.

## Architecture Quality

The host diagnostic reference now mirrors the generated CUDA task-body branch:
shape-field logits use the tiled hidden-by-vocab formula, while scalar fallback
logits use the same `hidden[i%hidden_elements] * lm_head[i&3]` formula as the
device task. BF16 resident weights are decoded through the same dtype contract
used by `pto_cuda_tensor_arg_f32`.

Splitting the reference helpers keeps the resource-backed graph materializer
under the repository soft size target and makes the numerical checker easier
to review independently.

## Evaluation Run

Live A100 policy-length diagnostic capture:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-unit-math-live --run-submission-smoke \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1024 \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --token-cuda-live --kv-cuda-live --resident-cuda-live \
  --workspace-cuda-live --device 0 --arch compute_80 --repeat-runs 1 \
  --output-json tmp/cuda-backend/pto-serving-decode-loop-policy-sampled-ref-5948323b/qwen-decode-loop-runner.json
```

Result: MPK executed `1024/1024` decode steps, VDCores executed `64/64`,
both had zero scheduler errors, both observed device token feedback, and both
reported `diagnostic_logits_reference_status=pass` over `65536` checked logits
with `max_abs_error=0.0`.

Focused checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_data.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  -q -k 'diagnostic_logits_reference or diagnostic_logits_projection or diagnostic_logits_fallback or resource_backed_logits_summary or unit_math_full_rmsnorm or qwen_decode_loop'
```

## Remaining Gaps

This strengthens PTO persistent-device diagnostic correctness evidence. It is
still not a full-serving PTO row because the current task bodies remain
diagnostic formulas rather than full Qwen numerical kernels, and the paper
matrix still requires full-serving PTO, VDCores, and ThunderKittens rows.
