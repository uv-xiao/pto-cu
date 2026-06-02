# Qwen Preflight Aggregate

## Code And Data Changed

- Updated PTO Qwen full-serving preflight so
  `qwen_model_loader_or_token_loop` derives from detailed component checks
  instead of hard-failing on scaffold partial-stage labels.
- The aggregate now requires concrete evidence for prompt accounting, runtime
  input binding, CUDA token buffers, persistent decode args, token pointer
  table ownership, weight inventory, safetensors metadata, CUDA weight binding,
  persistent weight args/materialization, resident weight ownership, KV-cache
  binding, decode-loop runner, and generated Qwen task bodies.
- The current preflight remains `partial`, but now has one blocker:
  missing PTO Qwen/Qwen3-8B full-serving rows for `mpk_offline_decode` and
  `vdcores_offline_decode`.

## Architecture Quality

This narrows the full-serving blocker without promoting diagnostic evidence to
paper results. The preflight now distinguishes implemented model/token-loop
infrastructure from the real remaining paper gate: full-serving result rows
with correctness and latency/throughput metrics.

## Evaluation Run

Focused preflight validation passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py -q -k \
  pto_serving_preflight_captures_current_full_serving_gap
```

Current preflight capture under `tmp/` reports:

| Check | Status |
| ----- | ------ |
| `qwen_model_loader_or_token_loop` | pass |
| `qwen3_8b_full_serving_rows_imported` | fail |
| Blocking gaps | 1 |

## Remaining Gaps

The remaining PTO paper blocker is unchanged: import full-serving
Qwen/Qwen3-8B persistent-device rows for both paper serving policies with full
numerical correctness and paper latency/throughput metrics.
