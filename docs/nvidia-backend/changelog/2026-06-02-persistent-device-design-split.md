# 2026-06-02 Persistent Device Design Split

## Code And Data Changed

- Converted `docs/nvidia-backend/persistent-device.md` into a short landing
  page.
- Moved the existing CUDA persistent-device design sections into focused files
  under `docs/nvidia-backend/persistent-device/`.
- Updated the paper-ready work-preparation read order to include the split
  design archive.

## Architecture Quality

The AICPU-gap and CUDA persistent-scheduler design remains reachable through
the original stable evidence path, while each focused design file stays below
the 300-line reviewability target.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
```

Result: passed.

## Remaining Gaps

This is a documentation-structure cleanup only. It does not add new CUDA
persistent-device runtime behavior or paper-ready benchmark rows.
