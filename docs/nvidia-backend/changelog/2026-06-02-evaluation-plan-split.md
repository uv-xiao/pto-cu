# 2026-06-02 Evaluation Plan Split

## Code And Data Changed

- Converted `docs/in_progress/nvidia_backend_paper_ready/evaluation_plan.md`
  into a concise landing page.
- Moved the detailed baseline, workload, hardware, metric, reproducibility,
  paper-output, and dispatcher-backlog sections into focused files under
  `docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/`.
- Updated shared contracts, work-preparation read order, and goal-progress
  source evidence to reference the split structure.

## Architecture Quality

- Keeps the paper-ready evaluation plan under the repo's reviewability target
  without removing baseline coverage for MPK, VDCores, CUDA Graph, cuBLAS,
  vLLM, SGLang, ThunderKittens, A100, or H200.
- Corrects the documentation contract for the sharded capture-import viewer
  collection.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability cleanup only. It does not add final paper-grade raw
captures or close the remaining work-queue items.
