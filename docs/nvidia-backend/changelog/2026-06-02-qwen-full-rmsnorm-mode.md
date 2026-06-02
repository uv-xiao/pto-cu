# Qwen Full RMSNorm Mode

## Code And Data Changed

- Added `--resource-backed-numeric-task-mode unit_math_full_rmsnorm`.
- The new mode keeps the existing resource-backed unit-math branches, but
  passes `scalar_arg_count == 1` for `qwen_rmsnorm_input` so the generated CUDA
  task body runs its full hidden-buffer RMSNorm reduction branch.
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

This exposes the generated full RMSNorm reduction branch, but it is still
diagnostic. PTO still needs full hidden-size Qwen kernels, real full-serving
execution, and viewer rows for `mpk_offline_decode` and `vdcores_offline_decode`.
