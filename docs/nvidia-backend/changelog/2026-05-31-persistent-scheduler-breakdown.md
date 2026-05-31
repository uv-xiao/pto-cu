# 2026-05-31 Persistent Scheduler Breakdown

## Code And Data Changed

- Added `cuda_scheduler_breakdown.py` to summarize persistent-device smoke
  JSON into reviewable scheduler breakdown JSON, Markdown, and SVG reports.
- Captured A100 and H200 `graph_descriptor_layered_cross` persistent-device
  smokes with three scheduler blocks, four worker blocks, and 20 repeat runs.
- Added the generated breakdown artifact under
  `tmp/cuda-backend/scheduler-breakdown-6f7a1040/` to the persistent-device
  paper-evaluation evidence refs.
- Regenerated `paper_readiness_audit.json`.

## Architecture Quality

The breakdown report is derived from runtime-owned smoke fields instead of
new runtime behavior. It separates scheduler ready-queue counters, worker
task-execution timing, and host synchronization overhead so reviewers can see
which part of the CUDA persistent-device flow is being measured.

## Evaluation Run

The fixture-based pytest first failed because `cuda_scheduler_breakdown.py`
did not exist. After implementation, the paired A100/H200 layered-cross smoke
passed validation with zero device scheduler errors, expected dispatch IDs,
expected graph topology, and 20 repeat launches per GPU.

## Remaining Gaps

- MPK persistent-kernel and VDCores queue/resource-policy runs still need
  matching workload metadata and viewer import.
- LLM serving and broader ThunderKittens sweeps remain separate paper-readiness
  blockers.
