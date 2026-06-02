# Qwen Layer KV Layout

## Code And Data Changed

- Preserved `layer_index` from Qwen weight descriptors through resident-weight
  materialization and graph materialization summaries.
- Carried each layer index in `scalar_args[3]` for non-logits layer tasks.
- Added `out_batch_stride` to the Qwen submission plan as the KV-cache batch
  stride used by generated task bodies.
- Sized mutable KV cache storage as float32 while retaining BF16 as the model
  compute dtype in the lifecycle plan.
- Updated generated QKV, QK-normalized K writeback, and attention read code to
  index KV as `[layer][batch][token][kv_head][head_dim]`.
- Kept the A100 run artifact under
  `tmp/cuda-backend/qwen-layer-kv-layout-mpk-2026-06-03/`.

## Architecture Quality

The KV allocation plan already described a layer-partitioned cache layout, but
the generated CUDA task bodies only indexed batch, token, KV head, and head
dimension. That let different Qwen layers share the same live KV cache region.

This change makes the implementation match the cache layout contract and the
current `float *c` / `float *d` persistent task ABI. It is a prerequisite for
prompt prefill because prefilled per-layer K/V state must not be overwritten,
read from another layer's cache region, or stored in an allocation sized for a
different element type.

## Evaluation Run

Focused unit verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `57 passed`.

The real A100 resource-backed smoke also passed:

```bash
ARTIFACT=tmp/cuda-backend/qwen-layer-kv-layout-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 900 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 4 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --cache-root "$ARTIFACT/cache" --device 0 --arch compute_80 \
  --output-json "$ARTIFACT/qwen-decode-loop-runner.json"
```

Result: `resource_backed_execution.status=pass`, 10 scheduled tasks completed,
and scheduler error count stayed zero.

The stronger full-prefix A100 run also passed:

```bash
ARTIFACT=tmp/cuda-backend/qwen-layer-kv-layout-full-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 1200 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection prefix \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --cache-root "$ARTIFACT/cache" --device 0 --arch compute_80 \
  --output-json "$ARTIFACT/qwen-decode-loop-runner.json"
```

Result: `resource_backed_execution.status=pass`, all 255 materialized Qwen
tasks completed, and scheduler error count stayed zero. The artifact records
`layer_0_attention_qkv.layer_index=0`, `layer_35_attention_qkv.layer_index=35`,
task shape fields `a_batch_stride=64`, `b_batch_stride=1088`,
`out_batch_stride=16`, and `key_cache.byte_count=2566914048` with
`element_dtype=float32`.

## Remaining Gaps

This slice does not implement real prompt prefill. Full Qwen numerical
correctness still requires prefilled per-layer KV state for prompt tokens and
remaining task-math fidelity fixes.
