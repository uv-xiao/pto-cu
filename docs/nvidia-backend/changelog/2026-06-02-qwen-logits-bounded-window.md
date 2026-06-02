# Qwen Logits Bounded Window

## Code And Data Changed

- Added `qwen_logits.task_shape_fields.scalar1` as the active diagnostic
  vocabulary width emitted by the Qwen weight-argument shape contract.
- Updated the generated `qwen_logits` persistent-device task body to compute
  only the active diagnostic logits window while preserving `task->cols` as
  the true model vocabulary stride.
- Tightened weight-argument artifact freshness so stale logits descriptors
  without positive `cols`, `inner`, and `scalar1` are regenerated.
- Updated the resource-backed logits summary to report the bounded diagnostic
  write extent instead of treating the full logits allocation as written.
- Added `--resource-backed-logits-active-cols` so focused live evaluation can
  request either a wider positive active-vocabulary window or the descriptor's
  full `cols` extent without generating a second `qwen_logits` task body.
  The policy lives in
  `examples/cuda/qwen_decode_loop_runner_impl/logits_active_cols.py`.
- Recorded the applied active-logits-column policy in resource-backed raw
  execution results and viewer-import statistics.

## Architecture Quality

The diagnostic bound is now an explicit task descriptor contract instead of an
implicit test assumption. The host packet, generated device task body, artifact
loader, and live-result summary all use the same `scalar1` field, making the
current partial-vocabulary Qwen path reviewable without claiming full-serving
coverage.

The persistent-device callable remains one generated `qwen_logits` task. This
does not introduce a separate host-launched kernel form, and it leaves the
future full-vocabulary tensor-core logits implementation as a clear replacement
for this diagnostic path.

The active-column override changes the host packet descriptor for
`qwen_logits` only. It does not mutate resident descriptor artifacts, and it
does not change the device function ABI. This keeps descriptor-default
evidence comparable while allowing one-off evaluation runs to prove whether
full-vocabulary scalar logits execution is viable for a selected workload.

## Evaluation Run

Focused regressions passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q -k \
  'generated_source_contains_qwen_unit_math_kernels or weight_args_loader_rejects_shape_field_stale_artifact or qwen_weight_descriptors_emit_callable_shape_fields or launch_packet_carries_cuda_task_shape_fields or active_logits_written_elements_uses_diagnostic_window or resource_backed_logits_summary_marks_partial_vocab_coverage'
```

Generated CUDA source evidence was written under
`tmp/cuda-backend/qwen-logits-bounded-window/`. The source includes
`active_logits_cols`, `active_logits_elements`, and the
`token < active_logits_cols` feedback reduction.

A bounded resource-backed live smoke was attempted with:

```bash
PYTHONPATH=$PWD:$PWD/python timeout 240 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-repeat-runs 1 --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-logits-bounded-window/qwen-decode-loop-runner-live.json
```

The command timed out after 240 seconds and did not produce a completed JSON
result.

Focused regressions for the active-column override passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q -k \
  'logits_active_cols_override or qwen_weight_descriptors_emit_callable_shape_fields or launch_packet_carries_cuda_task_shape_fields or build_execution_result'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
```

## Remaining Gaps

This change makes the current Qwen logits path honest and bounded for
diagnostics, but it is not a paper-ready full-vocabulary serving result. The
remaining implementation gap is a full-vocabulary logits path backed by tiled
or tensor-core CUDA math that can complete the resource-backed smoke and then
scale to the paper evaluation matrix.
