# Serving Command Plan Shards

## Code And Data Changed

- Replaced the committed `serving_command_plan.json` payload with
  `data/serving_command_plan/index.json` and one record per serving command
  plan.
- Extended the shared viewer-data sharding helper for
  `serving_command_plans`.
- Updated the HTML viewer, command-plan generator, and focused review test to
  load the command plan through the logical sharded collection path.

## Architecture Quality

Each planned serving run now has a small standalone JSON file. Reviewers can
inspect one baseline/workload/batch command without scrolling through the full
serving plan.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability cleanup only. The paper-readiness work queue still
controls which serving rows must be captured and imported.
