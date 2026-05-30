# 2026-05-31 Persistent Baseline Run Contracts

## Code And Data Changed

- Added `mpk_persistent_scheduler_trace` to the paper-baseline run data.
- Added `vdcores_resource_policy_trace` to the paper-baseline run data.
- Regenerated `paper_readiness_audit.json` so the persistent-device claim now
  points at planned MPK and VDCores run records instead of reporting missing
  run contracts.
- Tightened viewer-data validation and focused review tests to require these
  run IDs.

## Architecture Quality

The persistent-device paper claim now has explicit baseline-run contracts for
the two systems it is meant to compare against. This separates scheduler and
resource-policy evidence from the LLM serving throughput rows, making the
planned artifacts and importer requirements easier to review.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
    --output docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json
```

The generated audit still reports `not_paper_ready`, but the
`persistent_device_scheduler_overhead` blockers now identify the planned MPK
and VDCores run records that must be captured and imported.

## Remaining Gaps

- The new run records are planned contracts. They do not contain raw MPK or
  VDCores scheduler traces yet.
- The next paper-evaluation slice must execute these commands on H200-class
  hardware, normalize the raw JSON, and import viewer result rows.
