# 2026-06-03 Qwen Logits Row Reference

## Code And Data Changed

- Updated resource-backed Qwen logits diagnostic-reference sampling so large
  vocab buffers sample across output rows instead of checking only row 0.
- Added `checked_row_count` to the raw diagnostic logits reference and compact
  resource-backed viewer-row statistic.
- Kept benchmark-viewer data unchanged; live evidence remains under `tmp/`.

## Architecture Quality

This strengthens the current diagnostic Qwen execution evidence for batched
serving policies. The runner already copies the full logits buffer for focused
MPK-policy checks; the host-side projection reference now spans all checked
batch rows within the bounded reference budget, so row-specific hidden and
weight indexing bugs are easier to catch before full-serving promotion.

This is still diagnostic evidence, not full Qwen token-level numerical
correctness.

## Evaluation Run

- Failed before the helper fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_single_context_session.py \
    -q -k diagnostic_logits_reference_samples_large_vocab_rows
  ```

- Passed focused resource-backed importer coverage:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py \
    -q -k resource_backed_importer_emits_diagnostic_rows
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-logits-row-reference-mpk-2026-06-03
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
  16 batch rows, and passed with `max_abs_error=2.3e-07`.

## Remaining Gaps

PTO still needs full Qwen token-level numerical comparison and full-serving
MPK/VDCores rows before the LLM-serving paper claim can pass.
