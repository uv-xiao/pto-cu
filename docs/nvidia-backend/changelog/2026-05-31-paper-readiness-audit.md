# 2026-05-31 Paper Readiness Audit

## Code And Data Changed

- Added `paper_readiness_audit.py` to generate a paper-claim readiness audit
  from the committed matrix, baseline-run, probe, and result JSON files.
- Added `paper_readiness_audit.json` to the benchmark viewer data set.
- Rendered the audit before the paper matrix in the HTML benchmark viewer.
- Extended the viewer-data validator and focused review tests so the committed
  audit must match regenerated output.

## Architecture Quality

The paper-readiness state is now generated from existing evidence instead of
being another hand-maintained status paragraph. Reviewers can see the blocker
list for each paper claim, while the validator prevents stale audit JSON after
matrix, probe, run, or result data changes.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
    --output docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json
```

The generated audit currently reports `not_paper_ready`, zero ready claims,
and four blocked claims.

## Remaining Gaps

- The audit is a readiness summary, not a performance result.
- The blocked claims still need the long MPK, VDCores, vLLM, SGLang,
  ThunderKittens, PTO, CUDA Graph, and vendor-library captures named in the
  matrix before any paper-ready performance claim can be made.
