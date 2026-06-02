# 2026-06-03 Qwen QK Norm Paged K-Cache

## Code And Data Changed

- Expanded the CUDA persistent-device task ABI from four to five tensor
  argument slots so one task can bind Q norm, K norm, RoPE cos, RoPE sin, and
  the runtime KV page table without hiding page-table state in an unrelated
  field.
- Updated `qwen_attention_qk_norm` descriptors to pass `kv_page_table` as
  `tensor_args[4]`.
- Updated generated QK-norm task source to map K-cache writeback through
  `kv_page_table[logical_page]`, falling back to identity mapping only when the
  runtime pointer is absent.
- Added `qwen_qk_norm_paged_k_cache_writeback_source` as a generated-source
  contract. Benchmark-viewer data was not changed; the live run is retained as
  raw evidence under `tmp/`.

## Architecture Quality

QK-norm now uses the same paged KV-cache address policy as QKV writeback and
attention-output reads. This removes a stale identity-page assumption from the
generated kernel path while keeping the descriptor-level evidence explicit for
human review.

## Evaluation Run

- Passed focused regression and ABI source checks:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py \
    -q -k 'rope_table_tensor_args or qwen3_shapes'

  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels

  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_persistent_codegen.py \
    -q -k generic_argument_slots
  ```

- Passed live A100 MPK-policy diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-qk-norm-paged-k-cache-mpk-2026-06-03
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

This is still a first-layer diagnostic. PTO still needs full Qwen numerical
comparison and full-serving MPK/VDCores rows before the LLM-serving paper claim
can pass.
