# 2026-06-02 Qwen RoPE Table Launch Binding

## Code And Data Changed

- Added activation-workspace-owned `rope_cos_table` and `rope_sin_table`
  runtime buffers for every Qwen serving workload.
- Initialized live diagnostic RoPE buffers as identity tables: cos `1.0f`,
  sin `0.0f`.
- Updated launch-packet preflight to resolve
  `device_ptr_source=runtime_buffers.*` tensor arguments into live
  `CudaPersistentDagTask::tensor_args` pointers.
- Added runner evidence symbol `qwen_rope_table_launch_packet_binding` and
  synchronized the example manifest, Qwen source map, and paper-readiness
  matrix wording.

## Architecture Quality

RoPE tables are now treated as runtime-owned device buffers rather than
resident model weights. Persistent weight materialization keeps the symbolic
descriptor slots, while the decode-loop launch path supplies the concrete live
CUDA pointers from the activation workspace. Identity initialization keeps
diagnostic launches deterministic without claiming full position-correct RoPE
table generation.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_graph_materialization.py -q`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'activation_workspace or decode_loop_runner or persistent_qwen_weight_materialization or task_decomposition'`.

## Remaining Gaps

PTO full-serving rows still require position-correct RoPE table population,
decode attention reduction, complete decode-loop execution, and viewer import
for both MPK-policy and VDCores-policy workloads.
