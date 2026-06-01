# 2026-06-01 Persistent Scheduler Claim Promotion

## Code And Data Changed

- Promoted `persistent_device_scheduler_overhead` in
  `paper_evaluation_matrix.json` from partial capture to
  `ready_for_paper_claim`.
- Added the imported MPK persistent scheduler trace JSON as an explicit raw
  artifact evidence reference.
- Removed the stale matrix missing-evidence item for VDCores scheduler
  diagnostics after confirming the VDCores run contract and result record keep
  diagnostic scheduler fields separate from the final guarded latency and
  correctness row.
- Regenerated paper-readiness audit, work queue, run-readiness, environment
  plans, and goal-progress JSON.
- Updated the NVIDIA review-artifact tests to expect two ready paper claims
  and three remaining work-queue items.

## Architecture Quality

The persistent-device scheduler-overhead claim now has a reviewable evidence
boundary:

- PTO persistent-device rows cover A100 and H200 `graph_layered_cross`
  captures.
- MPK contributes an H200 persistent scheduler trace with task registry,
  resource policy, dispatch trace, and scheduler slice breakdown.
- VDCores contributes a final guarded H200 latency/correctness row plus a
  separate queue/scheduler diagnostic row whose result metadata says it must
  not be treated as the final non-diagnostic latency row.

This keeps the CUDA persistent-device comparison honest while still allowing
the scheduler-overhead claim to be reviewed independently from the larger LLM
serving baseline matrix.

## Evaluation Run

Evidence inspected:

- `tmp/cuda-backend/scheduler-breakdown-6f7a1040/persistent-scheduler-breakdown-6f7a1040/`
- `tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json`
- `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-correctness-712f88e8/`
- `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-queue-scheduler-46872fa4/`

Verification commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

Result: generated review data with two ready claims and three remaining work
items.

## Remaining Gaps

- LLM serving remains blocked on paper-baseline result imports and the gated
  VDCores Llama run readiness.
- Tensor-core tile baselines remain blocked on official ThunderKittens sweep
  gaps.
