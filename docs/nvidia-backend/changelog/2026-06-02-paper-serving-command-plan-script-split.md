# 2026-06-02 Paper Serving Command Plan Script Split

## Code And Data Changed

- Replaced
  `.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`
  with a short CLI and compatibility export layer.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan_impl/`
  for plan assembly, baseline command builders, JSON I/O, path helpers,
  record identity helpers, shell rendering, and Git metadata.
- Preserved public helper imports such as `build_plan`,
  `COMMAND_BUILDERS`, `vllm_commands`, `sglang_commands`, `mpk_commands`,
  `vdcores_commands`, and `thunderkittens_commands`.
- Updated the `llm_serving_decode` benchmark evidence references to point at
  the new vLLM and SGLang command-builder modules.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every new serving-command-plan module below the 300-line review
  target.
- Separates framework serving commands from kernel-family baseline commands,
  so MPK, VDCores, ThunderKittens, vLLM, and SGLang command policy can be
  reviewed independently.
- Keeps CLI behavior and generated `serving_command_plans` JSON unchanged for
  benchmark-viewer and run-readiness consumers.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py \
    .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan_impl/*.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_serving_command_plan_generates_policy_commands'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, focused serving-command-plan pytest, diff check,
  changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not add new serving captures. It improves the command-plan
  generator structure used to prepare paper-baseline serving runs.
