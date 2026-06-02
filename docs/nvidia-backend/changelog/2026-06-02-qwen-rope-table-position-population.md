# 2026-06-02 Qwen RoPE Table Position Population

## Code And Data Changed

- Added `rope_theta` to the Qwen3-8B lifecycle model shape from the mirrored
  config snapshot.
- Extended activation workspace plans with `rope_base_position`,
  `rope_theta`, and `rope_table_policy`.
- Replaced diagnostic identity RoPE table initialization with Qwen-theta
  cos/sin values for the workload `first_decode_position`.
- Added `qwen_position_rope_table_population` as separate runner evidence from
  launch-packet pointer binding.

## Architecture Quality

The runtime buffer owner now records both allocation and content policy for
RoPE tables. Dry-run pointer tables expose the same policy metadata as live
CUDA allocation, while live allocation also copies the computed cos/sin table
to device memory. Dynamic per-step refresh is still separated from this
first-position table population and remains part of full decode-loop execution.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_graph_materialization.py -q`.

## Remaining Gaps

PTO full-serving rows still require dynamic per-step RoPE refresh, decode
attention reduction, complete decode-loop execution, and viewer import for
both MPK-policy and VDCores-policy workloads.
