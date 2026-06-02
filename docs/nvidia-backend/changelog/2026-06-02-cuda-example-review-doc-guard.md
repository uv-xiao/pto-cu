# CUDA Example Review-Doc Guard

## Code And Data Changed

- Tightened `validate_cuda_examples.py` so each manifest example must map to
  one focused README-linked review doc.
- The guard checks review-doc line counts, maximum page length, benchmark id,
  runtime, method id, command, expected output, and script name.

## Architecture Quality

The NVIDIA examples now have a stronger human-review contract: executable
examples, manifest entries, and short per-example docs must move together.

## Evaluation Run

- `validate_cuda_examples.py` passed with the stricter review-doc checks.
- Focused benchmark-viewer, changelog, review-ready, pytest, and diff checks
  passed after the guard was added.

## Remaining Gaps

This preserves example/document alignment. It does not add new paper-grade
benchmark captures or close the remaining backend implementation gaps.
