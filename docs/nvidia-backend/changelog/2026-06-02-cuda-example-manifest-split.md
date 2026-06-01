# 2026-06-02 CUDA Example Manifest Split

## Code And Data Changed

- Replaced `examples/cuda/manifest.json` with a short `example_files` index.
- Added focused manifest shards under `examples/cuda/manifest/`:
  core examples, Qwen lifecycle examples, Qwen token/runtime examples, and
  Qwen weight examples.
- Updated `.agents/checks/validate_cuda_examples.py` to accept both the
  historical inline manifest and the sharded manifest index.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps the stable top-level manifest path used by the NVIDIA review guard.
- Keeps every CUDA example manifest shard below the 300-line review target.
- Preserves one validator entrypoint so reviewers do not need a new command
  to check example-to-benchmark and example-to-code evidence.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_cuda_examples.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  git diff --check
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'cuda_example_validator_passes or review_policy_changelog_and_examples_exist'
  ```

- Result: CUDA example validation, changelog validation, NVIDIA review guard,
  diff check, and focused pytest passed.

## Remaining Gaps

- This split does not add new CUDA examples or new benchmark measurements. It
  improves the reviewability of the existing example contract.
