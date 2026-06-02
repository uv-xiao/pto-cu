# 2026-06-03 Qwen Prompt Prefill Replay

## Code And Data Changed

- Added `--resource-backed-prefill-prompt` to
  `examples/cuda/qwen_decode_loop_runner.py`.
- Updated the resource-backed Qwen runner to replay the selected DAG once per
  active prompt token before bounded decode.
- Factored one packet submission path for both prompt prefill and decode, while
  keeping decode-token feedback disabled during prefill.
- Added result fields that report prefill status, expected prompt positions,
  executed prompt positions, and aggregate scheduler counters.

## Architecture Quality

The CUDA persistent-device path still uses one compiled DAG packet format. The
prefill replay updates RoPE and decode-position state per prompt token, then
the decode loop continues through the same task packet and prepared callable.
This avoids introducing separate `__global__` and `__device__` task variants.

## Evaluation Run

Focused Python tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `51 passed`.

A100 compact resource-backed Qwen prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prompt-prefill-first-layer-mpk-2026-06-03/
```

The artifact reports 18 prompt positions executed, 180 prefill tasks completed,
zero prefill scheduler errors, and a passing decode step. A heavier full-prefix
prefill run was started for stronger evidence but stopped after it exceeded the
convergence budget for this pass.

## Remaining Gaps

- Full-prefix prompt prefill should be re-run to completion.
- Full Qwen numerical correctness against the Hugging Face reference remains
  open before PTO rows can be promoted to paper-ready full-serving results.
