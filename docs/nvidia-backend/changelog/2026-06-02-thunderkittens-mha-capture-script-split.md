# 2026-06-02 ThunderKittens MHA Capture Script Split

## Code And Data Changed

- Replaced
  `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py`
  with a short CLI and compatibility export layer.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture_impl/`
  for GPU metadata, raw result records, shape execution, shape parsing, and
  latency statistics.
- Preserved public helper imports used by
  `thunderkittens_full_sweep_capture.py`, including `parse_shape`,
  `read_gpu_metadata`, and `run_shape`.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every new ThunderKittens MHA capture module below the 300-line review
  target.
- Separates device execution from result-record construction, so the serving
  proxy metrics and tensor-core tile metrics can be reviewed independently.
- Keeps the CLI command path stable for paper-baseline run-readiness records
  and serving command plans.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py \
    .agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture_impl/*.py \
    .agents/skills/cuda-backend-eval/scripts/thunderkittens_full_sweep_capture.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'thunderkittens_capture_builds_serving_decode_result or thunderkittens_full_sweep_capture_builds_importable_record'
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, focused ThunderKittens MHA pytest,
  benchmark-viewer data validation, diff check, changelog validation, and
  NVIDIA review guard passed.

## Remaining Gaps

- This split does not run new ThunderKittens H200 captures. It improves the
  capture wrapper structure used to generate those raw artifacts.
