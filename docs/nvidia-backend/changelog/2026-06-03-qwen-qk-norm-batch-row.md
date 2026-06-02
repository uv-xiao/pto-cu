# 2026-06-03 Qwen QK Norm Batch Row

## Code And Data Changed

- Updated generated `qwen_attention_qk_norm` source so the main Q/K region
  path derives `row` from `j / task->cols` for each element.
- Added `qwen_qk_norm_batch_row_index_source` to the Qwen task-body manifest
  and evidence symbol list.
- Kept benchmark-viewer data unchanged; live evidence remains under `tmp/`.

## Architecture Quality

This removes a batch correctness defect in the real Qwen QK-norm path. The
generated task now reads Q/K vectors from the current batch row and forms the
K-cache write index with that same row, instead of always using row 0.

The compact fallback diagnostic path remains single-row oriented; real Qwen
descriptors use the Q-width plus KV-width path that this change fixes.

## Evaluation Run

- Failed before the source fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-qk-norm-batch-row-mpk-2026-06-03
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

- Result: generated dispatch rebuilt (`cache_hit=false`), `mpk_offline_decode`
  completed 10 resource-backed task functions with zero scheduler errors,
  checked 2,430,976 logits elements, sampled token `63690`, and passed the
  diagnostic projection reference with `max_abs_error=1.2e-07`.

## Remaining Gaps

PTO still needs full Qwen numerical comparison and full-serving MPK/VDCores
rows before the LLM-serving paper claim can pass.
