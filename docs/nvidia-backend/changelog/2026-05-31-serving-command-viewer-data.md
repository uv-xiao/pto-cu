# 2026-05-31 Serving Command Viewer Data

## Code And Data Changed

- Added `serving_command_plan.json` to the benchmark-viewer data set.
- Extended the viewer to load and render a `Serving Commands` tab with
  per-baseline, per-policy, per-batch commands and raw artifact paths.
- Tightened `validate_benchmark_viewer_data.py` so the committed command plan
  must cover every LLM-serving paper-baseline run, serving policy, and batch
  size.
- Added review tests that require the committed command plan to cover MPK,
  VDCores, vLLM, SGLang, and ThunderKittens serving-family rows.

## Architecture Quality

The long-running H200 serving baseline work now has a versioned launch contract
instead of a private tmp-only command file. Reviewers can inspect the exact
model, prompt length, decode length, batch size, command line, and future raw
artifact path before any result is imported into the viewer.

## Evaluation Run

The command plan is generated from `serving_workloads.json` and
`paper_baseline_runs.json` with
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`. The
current committed plan has 35 rows: MPK 5, VDCores 5, vLLM 10, SGLang 10, and
ThunderKittens 5.

Focused viewer-data validation passed after the committed data file, viewer
rendering hook, and schema guard were added.

## Remaining Gaps

This is a launch-plan artifact, not a measured serving result. The LLM-serving
paper claim still needs the planned H200 baseline runs, raw JSON artifacts, and
viewer imports before promotion.
