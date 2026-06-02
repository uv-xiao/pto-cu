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
- Fixed host-side logits top-k summaries for multi-row logits buffers so
  `token_id` reports the row-0 vocabulary column, matching the device-side
  diagnostic greedy argmax feedback contract.

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

The full active-vocabulary policy completed for one VDCores-policy
resource-backed decode step:

```bash
PYTHONPATH=$PWD:$PWD/python timeout 420 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 10 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math --device 0 \
  --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-full-active-logits-2026-06-03/cache \
  --output-json tmp/cuda-backend/qwen-full-active-logits-2026-06-03/qwen-decode-loop-runner.json
```

The raw artifact under `tmp/cuda-backend/qwen-full-active-logits-2026-06-03/`
reported `status=pass`, 10 completed Qwen task functions, zero scheduler
errors, `logits_active_cols_policy.mode=full_descriptor_cols`,
`applied_scalar1_values=[151936]`, and `full_logits_buffer_checked` over
2,430,976 logits elements. The diagnostic projection reference checked 244
sampled elements with `max_abs_error=0.0`. One compact viewer row was imported;
the raw 249 KB JSON remains under `tmp/`.

The same full active-vocabulary policy also completed for one MPK-policy
resource-backed decode step after fixing the row-0 token-id summary:

```bash
PYTHONPATH=$PWD:$PWD/python timeout 420 .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 10 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols full \
  --resource-backed-numeric-task-mode unit_math --device 0 \
  --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-full-active-logits-mpk-2026-06-03-fixed/cache \
  --output-json tmp/cuda-backend/qwen-full-active-logits-mpk-2026-06-03-fixed/qwen-decode-loop-runner.json
```

The raw artifact under
`tmp/cuda-backend/qwen-full-active-logits-mpk-2026-06-03-fixed/` reported
`status=pass`, 10 completed Qwen task functions, zero scheduler errors,
`device_token_feedback_observed`, `full_logits_buffer_checked` over 2,430,976
logits elements, and diagnostic projection reference `max_abs_error=0.0`. One
compact viewer row was imported; the raw 249 KB JSON remains under `tmp/`.

Focused regressions for the multi-row host-summary fix passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q -k \
  'resource_backed_logits_summary or active_logits_written_elements'
```

## Remaining Gaps

This change makes the current Qwen logits path honest and bounded for
diagnostics, and now proves selected full active-vocabulary logits passes can
complete for both MPK and VDCores serving policies. It is still not a
paper-ready full-serving result: the remaining implementation gap is full Qwen
numerical correctness across the serving policy and import through the
full-serving viewer gate.
