# 2026-06-04 Qwen Full-Layer Prefill Vocab Status

## Code And Data Changed

- Made resource-backed Qwen workload status fail when a checked logits
  diagnostic reference reports `status=fail`.
- Added a focused result-assembly regression for checked diagnostic-reference
  failures.
- Captured full 36-layer prompt-prefill MPK evidence with bounded projections,
  including a full-vocabulary readout attempt.

## Architecture Quality

The resource-backed execution artifact now separates scheduler success from
numeric-check success. A CUDA packet can complete all tasks and still fail the
workload if the checked logits reference mismatches. This prevents failed
full-vocabulary diagnostics from being misread as passing paper-readiness
evidence.

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

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_workload_result_fails_failed_checked_logits_reference \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_resource_backed_execution_reports_task_coverage \
  -q
```

Result: both tests passed.

## Remaining Gaps

Full Qwen correctness remains open. The next model-runtime blocker is the
full-vocabulary logits reference mismatch after full-layer prompt-prefill
readout. PTO still needs full-vocab/full-projection model-equivalent token and
logit agreement against Hugging Face before MPK/VDCores full-serving rows can
be imported for paper claims.
