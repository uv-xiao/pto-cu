# Qwen Terminal Activation Workspace

## Code And Data Changed

- Added a shared activation-buffer count policy for resource-backed Qwen DAG
  packets.
- Workspace planning and live workspace validation now allocate one activation
  buffer per task when a packet ends before final readout.
- Terminal readout packets still use the logits buffer for `qwen_final_norm`
  and `qwen_logits`.
- Added focused coverage for non-readout terminal packet workspace sizing.

## Architecture Quality

This closes a packet-slicing correctness gap introduced by phase-specific
prompt prefill and readout packets. A prefix packet ending at an internal Qwen
task must keep its final hidden state in the activation chain; it cannot reuse
the logits buffer merely because it is the last task in that slice.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `56 passed`.

A100 prefix-64 prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prefix64-terminal-activation-2026-06-03/
```

The artifact reports 18 prompt positions, 1152 completed prefill tasks, a
64-task bounded decode packet, and zero scheduler errors.

## Remaining Gaps

- This is still diagnostic resource-backed execution, not paper-ready full
  Qwen numerical correctness against a Hugging Face reference.
- Current full-prefix prompt prefill remains too slow for paper-ready serving
  throughput.
