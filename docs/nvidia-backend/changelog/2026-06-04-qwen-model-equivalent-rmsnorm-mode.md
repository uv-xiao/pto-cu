# 2026-06-04 Qwen Model-Equivalent RMSNorm Mode

## Code And Data Changed

- Added `--resource-backed-numeric-task-mode model_equivalent` to
  `qwen_decode_loop_runner.py`.
- Made `model_equivalent` select the same full RMSNorm scalar contract as
  `unit_math_full_rmsnorm`, while reporting a distinct
  `resource_backed_model_equivalent_numeric_path` scope.
- Updated PTO paper-serving command rows so MPK and VDCores target runs request
  `model_equivalent` mode together with full QKV projection and full logits.
- Tightened benchmark-viewer data validation so PTO Qwen full-serving commands
  cannot omit `--resource-backed-numeric-task-mode model_equivalent`.

## Architecture Quality

The paper-target PTO commands no longer rely on the default diagnostic numeric
mode. That default kept RMSNorm tasks on an external-scale diagnostic bridge,
which is useful for smoke tests but is not a model-equivalent Qwen execution
path. The new mode makes the intent explicit without reusing the older
unit-math label for paper-serving commands.

## Evaluation Run

Focused TDD regression:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_model_equivalent_mode_selects_full_rmsnorm_reduction_branch \
  -q
```

Result: failed before the mode existed with
`ValueError: unknown numeric task mode: model_equivalent`, then passed after
the launch helper update.

One-layer live A100 comparison:

- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-attention-o/qwen-runner.json`
  used the old default diagnostic mode. It reported final-norm
  `max_abs_finite=3.213728`.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-attention-o/hf-layer1-norm-probe.json`
  recorded Hugging Face `model.norm(hidden_states[1])` at
  `max_abs_finite=43.34375`.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-full-rmsnorm/qwen-runner.json`
  used full RMSNorm and reported final-norm `max_abs_finite=43.317745`, with
  nonzero attention-O and a passing diagnostic logits reference.

Post-fix focused checks:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_model_equivalent_mode_selects_full_rmsnorm_reduction_branch \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_launch_packet_can_select_full_rmsnorm_reduction_branch \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_full_rmsnorm_mode_uses_full_hidden_vector_extent \
  -q

PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: three focused tests passed, and benchmark-viewer data validation
passed.

## Remaining Gaps

The one-layer scale now matches the Hugging Face layer-1 final-norm magnitude,
but the sample values and top tokens still differ. The next blocker is a
layer-0 math mismatch after full RMSNorm, likely in QK norm, attention, or MLP
semantics. Full Qwen token/logit agreement remains open.
