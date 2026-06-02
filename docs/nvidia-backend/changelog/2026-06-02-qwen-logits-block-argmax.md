# Qwen Logits Block Argmax

## Code And Data Changed

- Switched the generated `qwen_logits` persistent-device task body from
  element-threaded execution to block-threaded execution.
- The task now computes each full-vocabulary logit once, stores it in
  `task->out`, and uses shared-memory reduction over `task->out[token]` for
  device-side token feedback.
- Updated the Qwen task-body source-contract test to reject the previous
  duplicated `i == 0` full-vocabulary dot-product pass.

## Architecture Quality

The logits task remains a generated persistent-device task body, so the
orchestrator still emits one `qwen_logits` callable for the persistent DAG
instead of splitting the user-visible kernel into separate host-launch and
device-function forms.

This is an incremental runtime improvement: it removes duplicate argmax work
inside the existing scalar projection path, but does not pretend that scalar
full-vocabulary GEMV is the final serving design.

## Evaluation Run

Focused tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q -k \
  'qwen_weight_descriptors_emit_callable_shape_fields or materialized_weight_descriptor_preserves_task_shape_fields or launch_packet_carries_cuda_task_shape_fields'
```

Generated source evidence was written under
`tmp/cuda-backend/qwen-logits-block-argmax/`. The source includes
`pto_task_qwen_logits`, `logits_best_values`, and the
`token = threadIdx.x` reduction loop.

## Remaining Gaps

A bounded single-context resource-backed live smoke with the new task body
timed out at 240 seconds and was not imported as evaluation data. This change
removes duplicate argmax work, but the full Qwen projection is still a scalar
GEMV over the full vocabulary. Paper-grade full-serving evidence still needs a
tiled or tensor-core logits path.
