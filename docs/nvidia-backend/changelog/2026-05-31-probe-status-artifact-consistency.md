# 2026-05-31 Probe Status Artifact Consistency

## Code And Data Changed

Tightened `.agents/checks/validate_benchmark_viewer_data.py` so committed
per-machine probe status must match the raw A100/H200 probe JSON files it
references. The guard now loads each `latest_machine_status[*].artifact`,
finds the matching `paper_baseline_id`, and compares both `status` and
`blocking_gaps`.

Added a focused negative test that mutates a machine status away from the raw
probe artifact and verifies the validator rejects the drift.

## Architecture Quality

This removes a manual synchronization gap in the benchmark viewer. The viewer
can now render A100/H200 readiness status from committed JSON while the guard
proves those summaries are derived from the raw paired probe artifacts under
`tmp/`.

The change keeps planned artifacts and future runs separate from current
evidence: only statuses that agree with raw JSON can support current readiness
claims.

## Evaluation Run

The guard validates the existing paired probe artifact at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`.

The validation command is:

```bash
.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

The consistency guard prevents stale readiness summaries, but it does not
produce new baseline performance results. MPK, VDCores, vLLM, SGLang, and the
full ThunderKittens sweeps still need raw benchmark captures before their
paper-evaluation rows can be promoted.
