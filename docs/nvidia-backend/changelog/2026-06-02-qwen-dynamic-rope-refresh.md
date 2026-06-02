# 2026-06-02 Qwen Dynamic RoPE Refresh

## Code And Data Changed

- Added `refresh_rope_tables_for_decode_position` for resource-backed Qwen
  runtime buffers.
- Updated bounded decode-step execution to refresh live RoPE cos/sin tables for
  `first_decode_position + step_index` before each `run_prepared` call.
- Added `qwen_dynamic_rope_table_refresh` as a separate execution evidence
  contract.
- Updated the paper-readiness matrix, Qwen source map, and decode-loop runner
  docs so dynamic refresh is no longer listed as a PTO full-serving blocker.

## Architecture Quality

RoPE table ownership remains in the activation workspace, while resource-backed
execution owns per-step content refresh. This keeps pointer lifetime, table
contents, and graph submission separate enough for review: the launch packet
keeps stable tensor pointers, and only the buffer contents change as decode
positions advance.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_graph_materialization.py -q`.

## Remaining Gaps

PTO full-serving rows still require decode attention reduction, complete
decode-loop execution, and viewer import for both MPK-policy and VDCores-policy
workloads.
