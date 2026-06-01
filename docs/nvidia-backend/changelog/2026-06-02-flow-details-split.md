# 2026-06-02 Flow Details Split

## Code And Data Changed

- Converted `docs/nvidia-backend/flows.md` into a short landing page.
- Moved compile/launch, runtime-flow, Runtime-vs-Driver, TileLang JIT,
  lifecycle/memory/callable, design-consequence, and source sections into
  focused files under `docs/nvidia-backend/flows/`.
- Updated the paper-ready work-preparation read order to include the split
  flow archive.

## Architecture Quality

The CUDA compile/launch and runtime-flow comparison remains reachable through
the original stable entry point, while every focused flow file stays below the
300-line reviewability target.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
```

Result: passed.

## Remaining Gaps

This is a documentation-structure cleanup only. It does not change CUDA
runtime behavior or add new benchmark evidence.
