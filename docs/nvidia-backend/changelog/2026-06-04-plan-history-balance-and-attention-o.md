# 2026-06-04 Plan History Balance And Attention-O Scalar

## Code And Data Changed

- Updated `plan_history.json` to show the latest 12-commit work-focus window,
  recent slices, and a periodic reflection log that calls out excessive
  non-feature work.
- Rendered a compact focus bar, recent-slice table, and reflection-log table
  in the benchmark viewer summary.
- Tightened the benchmark-viewer validator so plan history must discourage
  sparse row-by-row tests, prefer large integrated tests, and record a
  reflection entry when tests/guardrails are taking too much time.
- Contracted the broad Python viewer artifact test by removing detailed
  plan-history policy assertions; the validator now owns those semantics.
- Fixed diagnostic-mode `qwen_attention_o` scalar args so
  `attention_o_projection_input_count` is preserved outside unit-math mode.

## Architecture Quality

The benchmark viewer is now a brief plan archive instead of a static note.
Reviewers can see whether recent work is runtime progress, tests/guardrails,
or viewer/docs without scanning the changelog. The reflection checkpoint asks
whether the next slice runs a benchmark model farther before adding more
reporting-only checks.

The test contraction keeps one broad viewer smoke plus one data-contract
validator. That avoids expanding the already large artifact test with
per-field plan-history assertions.

The Qwen scalar fix is model-runtime progress. Diagnostic `qwen_attention_o`
previously saw projection input count zero, so attention output could remain
zero even when the descriptor carried the correct 4,096-column projection
input count.

## Evaluation Run

Red check before updating data:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: failed with
`plan history test strategy must prefer large integrated tests`.

Post-fix checks:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py

node --check evaluations/nvidia/benchmark-viewer/viewer/viewer/render-summary.js

jq empty evaluations/nvidia/benchmark-viewer/data/plan_history.json
```

Result: all passed.

Focused Qwen scalar regression:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_attention_o_scalar_args_keep_projection_input_count_in_diagnostic_mode \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_projection_active_cols_override_targets_only_projection_callables \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_qwen_weight_descriptors_emit_callable_shape_fields \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_decode_step_state_extends_attention_o_kv_window \
  -q
```

Result: four tests passed.

Live A100 evidence:

- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-attention-o-scalar/qwen-runner.json`
  still reports zero attention-O because bounded 512-column QKV projection
  computes only Q and prunes K/V.
- `tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-full-qkv-attention-o/qwen-runner.json`
  reports nonzero QKV and attention-O activation with a passing diagnostic
  logits reference.

## Remaining Gaps

Full Qwen correctness remains open. Full 36-layer full-QKV/full-vocabulary
model-equivalent verification was started locally but did not finish in a
reasonable interactive window. The next useful slice should optimize or
bisect full-QKV correctness rather than add more reporting-only tests.
