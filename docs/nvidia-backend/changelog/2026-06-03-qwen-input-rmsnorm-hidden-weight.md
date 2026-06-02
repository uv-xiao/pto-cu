# 2026-06-03 Qwen Input RMSNorm Hidden Weight

## Code And Data Changed

- Fixed generated `qwen_rmsnorm_input` full-RMSNorm source so
  `input_layernorm_weight` is indexed by hidden column instead of `i & 3U`.
- Added `qwen_input_rmsnorm_hidden_weight_source` to the task-body manifest
  so the implementation contract is reviewable from generated-source evidence.
- Kept benchmark-viewer data unchanged; the live run below duplicates the
  existing MPK full-RMSNorm/full-logits diagnostic row and is retained only as
  raw implementation evidence under `tmp/`.

## Architecture Quality

This moves the generated Qwen kernels closer to real Qwen numerical semantics:
input RMSNorm, post-attention RMSNorm, and final RMSNorm now all use the
hidden-position norm weight for full-vector reductions. The path remains
diagnostic because the broader task bodies still need full-model numerical
comparison before any full-serving row can be imported.

## Evaluation Run

- Failed first, before the code fix:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed after the fix:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels

  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py \
    -q -k 'full_rmsnorm or launch_packet_can_select_full_rmsnorm'
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-input-rmsnorm-hidden-col-mpk-2026-06-03
  PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
    examples/cuda/qwen_decode_loop_runner.py --mode mock \
    --single-context-live-session --run-resource-backed-smoke \
    --resource-backed-task-selection first_layer_with_logits \
    --resource-backed-workload mpk_offline_decode \
    --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
    --resource-backed-worker-blocks 10 \
    --resource-backed-logits-check-policy final_step \
    --resource-backed-logits-active-cols full \
    --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
    --device 0 --arch compute_80 \
    --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Result: the generated dispatch was rebuilt (`cache_hit=false`),
  `mpk_offline_decode` completed 10 resource-backed task functions with zero
  scheduler errors, `full_logits_buffer_checked` over 2,430,976 logits
  elements, sampled token `63690`, and diagnostic projection reference
  `max_abs_error=1.2e-07`.

## Remaining Gaps

PTO still needs full Qwen numerical comparison against a reference model and
full-serving MPK/VDCores rows before the LLM-serving paper claim can pass.
