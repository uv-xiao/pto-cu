# 2026-06-04 Qwen Full-Layer Prefill Vocab Status

## Code And Data Changed

- Made resource-backed Qwen workload status fail when a checked logits
  diagnostic reference reports `status=fail`.
- Added a focused result-assembly regression for checked diagnostic-reference
  failures.
- Changed the host diagnostic logits reference to accumulate through float32,
  matching the generated CUDA logits kernel, and added mismatch-index metadata
  for failed checked comparisons.
- Captured full 36-layer prompt-prefill MPK evidence with bounded projections,
  including a passing full-vocabulary readout attempt.

## Architecture Quality

The resource-backed execution artifact now separates scheduler success from
numeric-check success. A CUDA packet can complete all tasks and still fail the
workload if the checked logits reference mismatches. This prevents failed
full-vocabulary diagnostics from being misread as passing paper-readiness
evidence.

The full-vocabulary mismatch was host-reference precision drift: the diagnostic
Python path accumulated logits in double precision while the generated CUDA
kernel accumulates in `float`. The diagnostic reference now rounds each
accumulation step through float32 instead of weakening the comparison gate.
Failed comparisons also report the first mismatching sampled logits index and
the max-error sampled index so future expensive full-vocabulary runs answer a
specific root-cause question.

## Evaluation Run

The bounded full-layer two-step MPK run wrote
`tmp/cuda-backend/qwen-prefill-layer36-mpk-2step-2026-06-04-rmsnorm-scale-fix/qwen-runner.json`.
Result: `prompt_prefill.status=prompt_prefill_executed`, 18 prompt positions,
4,554 prompt-prefill task completions, zero prompt-prefill scheduler errors,
two decode steps, a second full selected DAG with 255/255 completed tasks,
zero scheduler errors, nonzero bounded logits, diagnostic reference pass with
`max_abs_error=1.04e-06`, and device feedback tokens `[1647, 839]`.

The full-vocabulary readout rerun after the status gate wrote
`tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-status-gated/qwen-runner.json`.
Result: prompt prefill executed and scheduler counters stayed clean, but
`resource_backed_execution.status=fail` because the full-vocabulary checked
logits reference failed with `mismatch_count=80` and
`max_abs_error=2.793e-05`. The artifact still records full logits-buffer
coverage, all finite/nonzero logits, and sampled token `71590`.

The mismatch-diagnostic rerun wrote
`tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-mismatch-diagnostics/qwen-runner.json`.
Result: the first mismatch was sampled logits index `80`
(`-9.85181046` vs `-9.85178797`) and the max-error mismatch was sampled
logits index `234` (`-8.11010933` vs `-8.11008140`), which isolated the
remaining failure to host reference precision rather than buffer coverage or
task scheduling.

The post-fix full-vocabulary rerun wrote
`tmp/cuda-backend/qwen-prefill-layer36-mpk-full-logits-1step-2026-06-04-float32-reference/qwen-runner.json`.
Result: `resource_backed_execution.status=pass`, one workload passed, full
logits-buffer coverage, 2,430,976 finite/nonzero logits, top token `71590`,
and checked diagnostic reference `status=pass`, `mismatch_count=0`,
`max_abs_error=0` over 3,904 checked full-vocabulary elements.

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_workload_result_fails_failed_checked_logits_reference \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_resource_backed_execution_reports_task_coverage \
  -q
```

Result: both tests passed.

Focused verification for the logits diagnostic helpers:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_projection_accumulates_like_float32_kernel \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_reference_reports_mismatch_indices \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_reference_compares_tiled_vocab_projection \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_projection_checks_batch_rows \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_reference_samples_large_vocab_rows \
  tests/ut/py/test_nvidia_qwen_single_context_session.py::test_diagnostic_logits_reference_indices_stay_inside_active_window \
  -q
```

Result: six tests passed.

## Remaining Gaps

Full Qwen correctness remains open. The next model-runtime blocker is
model-equivalent token/logit agreement against Hugging Face after full-layer
prompt-prefill readout. PTO still needs full-vocab/full-projection
model-equivalent agreement before MPK/VDCores full-serving rows can be imported
for paper claims.
