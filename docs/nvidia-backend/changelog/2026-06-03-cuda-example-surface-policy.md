# CUDA Example Surface Policy

## Code And Data Changed

- Clarified `.agents/rules/example-requirements.md` so
  `examples/cuda/manifest.json` remains the source of truth for public CUDA
  examples.
- Clarified `examples/cuda/README.md` so narrow Qwen lifecycle probes stay
  support code for the advanced decode-loop example unless they become
  essential end-to-end cases.
- Removed local generated `__pycache__` directories from `examples/cuda/`.

## Architecture Quality

The public CUDA example surface stays small and reviewable while preserving
the implementation probes that currently support Qwen serving work.

## Evaluation Run

- `validate_cuda_examples.py` passed and continues to enforce the four-entry
  manifest and focused review docs.
- `git diff --check` passed.
- No benchmark data was added or removed in this policy-only change.

## Remaining Gaps

This does not move existing probe scripts or add new serving correctness
evidence. Full PTO serving correctness remains the next implementation gap.
