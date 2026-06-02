# Qwen MLP Oracle Correction

## Code And Data Changed

- Corrected the controlled proxy numeric oracle for `qwen_mlp_gate_up`.
- The oracle now matches the generated CUDA task body:
  `silu(gate_proj) * up_proj`.
- Added a focused regression assertion in
  `tests/ut/py/test_nvidia_qwen_task_body_math.py`.

## Architecture Quality

The task-body manifest is used as review evidence for generated CUDA source.
Keeping its Python oracle aligned with the generated device code prevents the
benchmark viewer and readiness reports from citing stale numeric evidence while
the PTO Qwen path moves toward real full-serving correctness.

## Evaluation Run

- The new assertion failed before the fix with stale oracle values
  `[10.5, 23.5, 38.5, 55.5]`.
- Focused task-body math tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_task_body_math.py`.
- Manifest regeneration passed and records corrected `qwen_mlp_gate_up`
  expected output `[0.365529, 2.642391, 7.144306, 13.748193]` under
  `tmp/cuda-backend/qwen-task-body-oracle-fix/qwen-task-bodies.json`.

## Remaining Gaps

This fixes stale numeric evidence for the controlled proxy. PTO still needs
full hidden-size Qwen kernels and real full-serving rows for
`mpk_offline_decode` and `vdcores_offline_decode`.
