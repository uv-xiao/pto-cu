# Qwen Prefill Readout Omission

## Code And Data Changed

- Added a prompt-prefill descriptor policy to
  `examples/cuda/qwen_decode_loop_runner_impl/resource_backed_execution.py`.
- Prompt prefill now reuses the selected Qwen task descriptors but omits
  readout-only `qwen_final_norm` and `qwen_logits` tasks.
- Resource-backed result JSON now reports the prompt-prefill graph task count
  and task policy.
- Added focused unit coverage for the descriptor policy and summary fields.

## Architecture Quality

This keeps the CUDA persistent-device task ABI unchanged. Prefill and decode
still use the same generated device task bodies and the same host task packet
format; only the per-phase descriptor subset changes. That matches the
serving-state requirement: prompt replay must populate layer KV state, while
final normalization and vocabulary projection are only needed for decode-token
selection.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `52 passed`.

A100 first-layer prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prefill-readout-omit-first-layer-mpk-2026-06-03/
```

The artifact reports `graph_task_count=8`, 18 prompt positions, 144 completed
prefill tasks, and zero scheduler errors. The following decode step completed
the full first-layer-with-logits packet with 10 tasks and zero scheduler
errors.

A100 bounded prefix-64 prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prefill-readout-omit-prefix64-mpk-2026-06-03/
```

The artifact reports `graph_task_count=64`, 18 prompt positions, 1152 completed
prefill tasks, and zero scheduler errors. The bounded decode packet also
completed 64 tasks with zero scheduler errors.

## Remaining Gaps

- Full-prefix prompt prefill still needs a completion artifact. It was started
  after this change but remained too slow for the convergence budget.
- This is still diagnostic resource-backed execution, not paper-ready full
  Qwen numerical correctness against a Hugging Face reference.
