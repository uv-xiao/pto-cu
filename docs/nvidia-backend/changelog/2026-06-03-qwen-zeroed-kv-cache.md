# Qwen Zeroed KV Cache

## Code And Data Changed

- Zero-initialized CUDA live KV-cache allocations for the Qwen decode-loop
  single-context allocator and standalone KV binding path.
- Added chunked device-copy zeroing with a 64 MiB host zero buffer so full
  cache allocations do not require one large host allocation.
- Exposed the KV-cache allocation and initialization policy in the
  decode-loop runner's `resource_lifecycle_policies` JSON field.
- Kept the A100 run artifact under
  `tmp/cuda-backend/qwen-zeroed-kv-cache-mpk-2026-06-03/`.

## Architecture Quality

The previous CUDA live KV-cache owner allocated full key/value cache buffers
without deterministic contents. The resource-backed attention path can scan
prompt-window cache positions, so the old behavior allowed uninitialized
prompt-history reads before true Qwen prefill exists.

This change does not claim prefill correctness. It makes the resource owner
deterministic and explicitly records that the full KV allocation is
zero-initialized but still lacks a prompt-prefill copy.

## Evaluation Run

Focused unit verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `48 passed`.

The real A100 resource-backed smoke also passed:

```bash
ARTIFACT=tmp/cuda-backend/qwen-zeroed-kv-cache-mpk-2026-06-03
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
and scheduler error count stayed zero. The artifact records
`allocation_policy=allocate_zeroed_full_kv_cache_without_prefill_copy` and
`initialization_policy.state=zero_initialized`.

## Remaining Gaps

Full Qwen numerical correctness still requires real prompt prefill into the KV
cache and remaining task-math fidelity fixes. This slice only removes
uninitialized KV state from the live resource path.
