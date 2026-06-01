# 2026-06-02 Qwen Persistent Weight Materialization Split

## Code And Data Changed

- Replaced the long `examples/cuda/qwen_persistent_weight_materialization.py`
  body with a short CLI/API wrapper.
- Added focused implementation modules under
  `examples/cuda/qwen_persistent_weight_materialization_impl/` for shared
  constants, input loading, ABI reflection, descriptor materialization, and
  manifest assembly.
- Added dispatch-log evidence for the refactor.

## Architecture Quality

- Keeps the existing `build_materialization_manifest` import path stable for
  decode-loop and resident-weight-table callers.
- Keeps each materialization implementation file below the 300-line review
  target while preserving the current JSON schema and manifest evidence
  symbols.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    examples/cuda/qwen_persistent_weight_materialization.py \
    examples/cuda/qwen_persistent_weight_materialization_impl/*.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_cuda_examples.py
  git diff --check
  PYTEST_K='persistent_qwen_weight_materialization_binds_resident_pointers or '
  PYTEST_K+='qwen_resident_weight_table_owner_materializes_during_lifetime or '
  PYTEST_K+='cuda_examples'
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k "$PYTEST_K"
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: CUDA example validation and review guard passed; focused pytest
  passed with `2 passed, 60 deselected`.

## Remaining Gaps

- This refactor does not add new CUDA benchmark results. It improves the
  reviewability of the Qwen persistent-device materialization example used by
  the paper-ready LLM serving path.
