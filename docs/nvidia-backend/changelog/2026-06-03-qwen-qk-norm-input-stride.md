# 2026-06-03 Qwen QK Norm Input Stride

## Code And Data Changed

- Updated `qwen_attention_qk_norm` generated CUDA source to use
  `a_batch_stride` for the input row base instead of using `lda`.
- Updated the QK-norm descriptor contract so `a_batch_stride` carries QKV's
  full `q_width + 2*kv_width` output row width.
- Added source and descriptor regressions for
  `qwen_qk_norm_qkv_input_stride_source`.
- Kept benchmark-viewer data unchanged; live evidence remains under `tmp/`.

## Architecture Quality

This separates QK-norm's input layout from its per-head math fields. `lda`
remains the head dimension used by RMSNorm/RoPE math, while `a_batch_stride`
now points row indexing at the producer QKV layout. Batched QK-norm rows
therefore read their own Q/K regions instead of stepping by head dimension.

## Evaluation Run

- Failed before the source and descriptor fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_qwen_weight_descriptors_emit_callable_shape_fields \
    -q
  ```

- Passed the broader focused Qwen suite:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py \
    tests/ut/py/test_nvidia_qwen_single_context_session.py \
    tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-qk-norm-input-stride-mpk-2026-06-03
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
    --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Result: `resource_backed_execution.status=pass`, 10 scheduler tasks
  completed with zero scheduler errors, sampled token `63690`, and diagnostic
  logits reference passed across 3,904 checked elements from 16 rows with
  `max_abs_error=1.195e-05` under tolerance `2e-05`.

## Remaining Gaps

PTO still needs full Qwen token-level numerical comparison and full-serving
MPK/VDCores rows before the LLM-serving paper claim can pass.
