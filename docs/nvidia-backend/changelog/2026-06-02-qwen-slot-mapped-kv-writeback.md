# Qwen Slot-Mapped KV Writeback

## Code And Data Changed

- Updated Qwen QKV persistent-device task bodies so key/value projection
  writeback uses runtime `kv_page_table` slots, decode position, and batch
  stride instead of flat row indexing.
- Bound `kv_page_table` into QKV descriptors as `tensor_args[3]` and carried
  the shared KV page size through QKV `scalar0`.
- Refreshed the Qwen task-body source map, example docs, and paper-readiness
  evidence summary so review data records slot-mapped KV writeback.
- Added a fresh task-body artifact under
  `tmp/cuda-backend/pto-serving-slot-mapped-kv-writeback-2026-06-02/`.

## Architecture Quality

This narrows the PTO Qwen full-serving gap by making QKV writeback follow the
same page-table abstraction already used by decode attention reads. The runner
now propagates decode position to task packets separately from the final
sampled-token output index, and the task body still supports logical-page
fallback when the runtime page table is absent.

## Evaluation Run

- Focused source and graph materialization tests passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_qwen_task_body_math.py
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q`.
- Generated task-body evidence:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python
  examples/cuda/qwen_persistent_task_bodies.py --output-json
  tmp/cuda-backend/pto-serving-slot-mapped-kv-writeback-2026-06-02/qwen-persistent-task-bodies.json
  --output-source
  tmp/cuda-backend/pto-serving-slot-mapped-kv-writeback-2026-06-02/qwen-persistent-task-bodies.cu`.

## Remaining Gaps

The PTO paper-readiness item still needs full-Qwen numerical correctness and
importable MPK-policy/VDCores-policy full-serving rows. Logits projection and
end-to-end token correctness remain the next Qwen semantic gates.
