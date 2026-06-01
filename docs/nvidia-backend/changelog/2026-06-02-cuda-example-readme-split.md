# 2026-06-02 CUDA Example README Split

## Code And Data Changed

- Converted `examples/cuda/README.md` into a short landing page with a review
  map.
- Moved each CUDA example description into a focused page under
  `examples/cuda/docs/`.
- Updated `.agents/checks/validate_cuda_examples.py` so manifest metadata is
  checked against the landing page plus split example docs.
- Kept the executable example sources in `examples/cuda/`; only the review
  documentation was split.

## Architecture Quality

The CUDA example artifact now has one stable entry point, short per-example
pages, and a guard that validates the split review surface. Each page keeps the
benchmark id, runtime id, method id, command, expected output, and caveat near
the executable example it documents.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
```

Result: passed.

## Remaining Gaps

This is a documentation-structure cleanup only. It does not add new CUDA
runtime behavior or new benchmark measurements.
