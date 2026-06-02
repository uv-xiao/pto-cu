# 2026-06-03 Qwen Attention Head-Dim Scale

## Code And Data Changed

- Updated generated `qwen_attention_o` source so bounded decode attention
  scores use `1 / sqrt(head_dim)` before the softmax max and weighted-value
  passes.
- Added `qwen_decode_attention_head_dim_scale_source` to the task-body
  manifest so reviewers can see this numerical contract in generated-source
  evidence.
- Kept benchmark-viewer data unchanged. The live run below duplicates the
  existing first-layer MPK diagnostic shape and is retained only as raw
  implementation evidence under `tmp/`.

## Architecture Quality

This moves the generated Qwen attention path closer to real scaled dot-product
attention while preserving the existing bounded diagnostic execution contract.
The attention body still does not prove full Qwen numerical correctness.

## Evaluation Run

- Failed before the source fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-attention-scale-mpk-2026-06-03
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
