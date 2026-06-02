# 2026-06-02 Qwen Unit Math Normal Graph

## Code And Data Changed

- Routed `examples/cuda/qwen_unit_math_live_impl/graph.py` through
  `simpler_setup.cuda_normal_graph` for task fan-in, dependent spans, and
  dependent-array construction.
- Kept `make_tasks()` as a compatibility wrapper while exposing
  `make_graph_arrays()` for runner code that needs all graph descriptor
  arrays.
- Updated `examples/cuda/qwen_unit_math_live_impl/runner.py` so repeat
  launches reset fan-in from the lowered graph instead of a handwritten
  constant.
- Added a focused unit test proving the Qwen unit-math DAG lowers to the
  expected `0 -> 1 -> 2 -> 3` task chain.

## Architecture Quality

The Qwen unit-math live example now uses the same normal-graph lowering
boundary as CUDA scene tests and persistent smoke. This removes another
handwritten DAG descriptor from example code while keeping task-body binding
inside the example-specific module.

## Evaluation Run

- Passed focused check:
  `.venv/bin/python -m py_compile examples/cuda/qwen_unit_math_live_impl/graph.py examples/cuda/qwen_unit_math_live_impl/runner.py`
- Passed focused check:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_unit_math_live.py -k unit_math_live_graph`
- Passed guard checks:
  `.venv/bin/python .agents/checks/validate_nvidia_changelog.py`
  and `.venv/bin/python .agents/checks/check_nvidia_review_ready.py`.

## Remaining Gaps

- This is graph descriptor evidence only. Full Qwen decode remains tracked by
  the existing serving work queue and paper-readiness gap reports.
