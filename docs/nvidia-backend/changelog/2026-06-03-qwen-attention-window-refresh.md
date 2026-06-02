# 2026-06-03 Qwen Attention Window Refresh

## Code And Data Changed

- Updated `set_decode_step_state` so resource-backed decode submissions refresh
  `qwen_attention_o.inner` to `decode_position + 1`.
- Added a regression test proving the attention-output task receives the
  dynamic KV window while neighboring tasks only receive scalar decode-position
  state.
- Updated the Qwen decode-loop runner doc and CUDA example README line count.

## Architecture Quality

`qwen_attention_o` uses `CudaPersistentDagTask::inner` as the softmax KV scan
window. The descriptor supplies an initial shape, but repeated decode steps
need a per-step window so attention includes the current token just written to
the paged KV cache. The refresh is scoped to function id `7104`, leaving logits
and other task-local `inner` meanings unchanged.

## Evaluation Run

Failed before the code change, then passed after it:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q \
  -k decode_step_state_extends_attention_o_kv_window
```

Focused related verification also passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q \
  -k 'decode_step_state or qwen_weight_descriptors_emit_callable_shape_fields'
```

Result: `2 passed, 25 deselected`.

Live A100 MPK-policy diagnostic also passed:

```bash
ARTIFACT=tmp/cuda-backend/qwen-attention-window-refresh-mpk-2026-06-03
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

Result: `resource_backed_execution.status=pass`, 10 scheduler tasks completed
with zero scheduler errors, sampled token `63690`, and diagnostic logits
reference passed across 3,904 checked elements from 16 rows with
`max_abs_error=1.195e-05` under tolerance `2e-05`.

The same A100 MPK-policy diagnostic was then rerun for two decode steps with
full active logits checked on every step:

```bash
ARTIFACT=tmp/cuda-backend/qwen-attention-window-refresh-multistep-mpk-2026-06-03
PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 2 \
  --resource-backed-worker-blocks 10 \
  --resource-backed-logits-check-policy every_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 --cache-root $ARTIFACT/cache \
  --output-json $ARTIFACT/qwen-decode-loop-runner.json
```

Result: both materialized decode steps passed. Each step completed 10
persistent tasks with zero scheduler errors, committed the device-selected
token back into the next-step input slot, and passed the diagnostic logits
reference across 3,904 checked elements with `max_abs_error=1.195e-05`.
Step 0 selected token `63690`; step 1 selected token `48084`.

## Remaining Gaps

This fixes the per-step launch-packet window for resource-backed diagnostics.
Full Qwen token-level correctness and full-serving MPK/VDCores rows remain
open before the LLM-serving paper claim can pass.
