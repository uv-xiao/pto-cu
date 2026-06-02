# Qwen Full RMSNorm Mode

## Code And Data Changed

- Added `--resource-backed-numeric-task-mode unit_math_full_rmsnorm`.
- The new mode keeps the existing resource-backed unit-math branches, passes
  `scalar_arg_count == 1` for `qwen_rmsnorm_input`, and caps the diagnostic
  hidden-stage packet extent to one 4,096-element hidden vector so the current
  generated O(n²) RMSNorm branch remains runnable.
- Preserved the existing `unit_math` mode and its external-scale RMSNorm
  bridge for already captured diagnostic artifacts.

## Architecture Quality

The resource-backed decode-loop runner can now select between the external
RMSNorm scale bridge and the generated full-reduction path explicitly. This
makes the numerical-correctness ladder reviewable in raw artifacts and result
contracts instead of hiding the branch choice inside task arguments.

## Evaluation Run

- Focused packet tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py`.
- Python compile passed for the decode-loop runner and touched helper modules.

## Remaining Gaps

This exposes the generated full RMSNorm reduction branch, but it is still a
bounded diagnostic. PTO still needs numerically correct full hidden-size Qwen
kernels, real full-serving execution, and viewer rows for `mpk_offline_decode`
and `vdcores_offline_decode`.
