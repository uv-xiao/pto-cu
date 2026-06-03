# 2026-06-03 Qwen Attention Batch Rows

## Code And Data Changed

- Removed callable-local `rows=query_heads` overrides from
  `qwen_attention_qk_norm` and `qwen_attention_o` descriptors.
- Updated generated QK-norm source to derive query-head count from
  `(cols - kv_width) / head_dim`.
- Updated generated attention-output source to derive query-head count from
  `cols / head_dim`.
- Added descriptor and generated-source regressions so launch-packet `rows`
  continues to represent workload batch rows for attention tasks.

## Architecture Quality

The prior descriptor contract overloaded `rows` as query-head count for
attention tasks. The resource-backed runner also uses `rows` from the workload
plan to size task extents and to form batch-local KV-cache bases. Overriding it
with query-head count made attention tasks execute 32 rows for Qwen/Qwen3-8B
even when the serving workload batch size was 16. Keeping `rows` as batch size
and deriving query-head count from width/head fields aligns task extents,
activation-buffer sizing, and KV-cache row indexing.

## Evaluation Run

Focused regressions first failed on the stale `rows` descriptor fields and old
generated source, then passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  -q -k 'test_qwen_weight_descriptors_emit_callable_shape_fields or test_generated_source_contains_qwen_unit_math_kernels'
```

Result: `2 passed, 44 deselected`.

Adjacent Qwen descriptor, task-body, decode-loop, and single-context tests
passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -q
```

Result: `68 passed`.

Generated Qwen task-body source compiled to PTX:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-attention-batch-rows/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-attention-batch-rows/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-attention-batch-rows/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-attention-batch-rows/qwen-persistent-task-bodies.ptx
```

A full-prefix, full-logits A100 MPK-policy run completed all 255 selected
tasks with zero scheduler errors:

```text
tmp/cuda-backend/qwen-attention-batch-rows-active-prompt-mpk-2026-06-03/qwen-decode-loop-runner.json
```

The artifact still reports empty row-0 top-k and a diagnostic logits reference
failure with `max_abs_error = 4.198e-05` over 3,904 checked elements. Treat it
as scheduler/task-shape evidence only; it does not prove Hugging Face token or
logit agreement.

## Remaining Gaps

This fixes a task-shape fidelity bug but does not close full Qwen numerical
correctness. The refreshed full-prefix artifact still has non-finite row-0
logits, so the next kernel-fidelity step is to localize which upstream hidden
row first becomes non-finite before final norm/logits. PTO rows still need
token/logit agreement against the Hugging Face Qwen/Qwen3-8B reference plus
latency and throughput metrics for both MPK and VDCores serving policies.
