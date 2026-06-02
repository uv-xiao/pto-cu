# 2026-06-03 Qwen Attention Batch KV Read

## Code And Data Changed

- Updated generated `qwen_attention_o` source so projection and non-projection
  decode-attention reads include a batch-local KV-cache base derived from
  `row`, `b_batch_stride`, `kv_heads`, and `head_dim`.
- Added `qwen_attention_o_batch_local_kv_read_source` as an explicit manifest
  contract and a source-level regression that verifies all four softmax KV read
  sites use the shared `kv_read_base`.
- Kept benchmark-viewer data unchanged; live evidence remains under `tmp/`.

## Architecture Quality

This aligns `qwen_attention_o` KV-cache read indexing with the existing QKV
and QK-norm writeback layout. Batched decode rows now read their own
sequence-capacity slice instead of reusing row-zero key/value entries.

The change is source-local in the generated task body and keeps the scheduler,
launch packets, and viewer data unchanged.

## Evaluation Run

- Failed before the source fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py -q
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
  ARTIFACT=tmp/cuda-backend/qwen-attention-batch-kv-read-mpk-2026-06-03
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
