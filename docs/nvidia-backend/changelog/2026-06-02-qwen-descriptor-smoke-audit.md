# 2026-06-02 Qwen Descriptor Smoke Audit

## Code And Data Changed

- Added the Qwen descriptor-smoke raw artifact to the LLM serving paper
  evaluation matrix.
- Added the descriptor-smoke viewer result requirement for the A100
  `pto_persistent_device` row.
- Regenerated `paper_readiness_audit.json` and
  `paper_readiness_work_queue.json` from the current matrix.

## Architecture Quality

The paper-readiness queue now distinguishes the executed descriptor-function
smoke from the still-missing resource-backed full Qwen decode loop. This keeps
diagnostic CUDA evidence reviewable without promoting it to full-serving
evidence.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/\
refresh_nvidia_review_artifacts.py
```

Result: regenerated audit/work-queue data reports 37 LLM-serving raw
artifacts and 10 viewer rows, while keeping four paper-readiness work items.

## Remaining Gaps

- Execute and import resource-backed full Qwen decode-loop rows for PTO.
- Resolve the VDCores Qwen3-8B full-serving correctness blocker.
- Import ThunderKittens-family full-serving Qwen3-8B rows.
