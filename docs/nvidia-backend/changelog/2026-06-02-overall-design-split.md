# 2026-06-02 Overall Design Split

## Code And Data Changed

- Replaced `docs/nvidia-backend/overall.md` with a short stable landing page.
- Added focused design pages under `docs/nvidia-backend/overall/` for
  architecture/scope, runtime shape, build/kernel contracts,
  semantics/testing/roadmap, and sources.
- Updated `docs/in_progress/nvidia_backend_paper_ready/work_preparation.md`
  to include the focused overall-design files in the review read order.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps the stable `overall.md` link used by README and reviewer guidance.
- Keeps every overall-design page below the 300-line review target.
- Separates historical design contracts by reader intent so future updates can
  touch runtime shape, build contracts, or roadmap details independently.

## Evaluation Run

- Focused validation passed:

  ```bash
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: changelog validation and NVIDIA review guard passed.

## Remaining Gaps

- This split does not add runtime code or benchmark data. It improves the
  reviewability of the CUDA backend design entrypoint.
