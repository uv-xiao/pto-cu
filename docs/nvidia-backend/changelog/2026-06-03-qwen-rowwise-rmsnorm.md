# 2026-06-03 Qwen Row-Wise RMSNorm

## Code And Data Changed

- Updated generated `qwen_rmsnorm_input`, `qwen_rmsnorm_post_attention`, and
  `qwen_final_norm` source so full RMSNorm branches reduce each batch row over
  descriptor `cols`.
- Updated RMSNorm norm-weight indexing to use the hidden column instead of a
  flattened batch element index in full and external-scale paths.
- Added `qwen_rowwise_rmsnorm_batch_source` as an explicit task-body manifest
  contract.
- Kept benchmark-viewer data unchanged; live evidence remains under `tmp/`.

## Architecture Quality

This removes a batch correctness defect from the generated Qwen task bodies.
The previous block RMSNorm path computed one scale over `task->n`, which
blended all rows in a batched serving packet. The generated code now computes
the row from `j / task->cols`, derives `row_base`, reduces over the row's
hidden columns, and applies column-local norm weights.

The generated implementation is still diagnostic and not a full Qwen serving
claim, but it moves the resource-backed path closer to the real model
semantics required for full-serving promotion.

## Evaluation Run

- Failed before the source fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed the focused related suite:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    tests/ut/py/test_nvidia_qwen_single_context_session.py \
    tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-rowwise-rmsnorm-mpk-2026-06-03
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

- Result: `mpk_offline_decode` completed 10 resource-backed task functions
  with zero scheduler errors, copied and checked the full 2,430,976-element
  logits buffer, sampled token `63690`, checked 3,904 projected logits across
  16 batch rows, and passed with `max_abs_error=1.195e-05` under a `2e-05`
  diagnostic projection tolerance.

## Remaining Gaps

PTO still needs full Qwen token-level numerical comparison and full-serving
MPK/VDCores rows before the LLM-serving paper claim can pass.
