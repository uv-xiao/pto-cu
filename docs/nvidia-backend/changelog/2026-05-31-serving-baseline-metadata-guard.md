# 2026-05-31 Serving Baseline Metadata Guard

## Code And Data Changed

- Tightened `validate_benchmark_viewer_data.py` so every
  `llm_serving_paper_baselines` run requires `model_and_prompt_shape` and
  `batch_or_concurrency_policy`.
- Updated MPK, VDCores, vLLM, and SGLang serving run contracts to include the
  missing required metrics.
- Regenerated paper-baseline run-readiness data with the current branch commit
  and refreshed the paper-readiness audit.
- Added a focused regression test proving incomplete LLM-serving run metadata
  is rejected before import.

## Architecture Quality

LLM-serving comparisons now have an explicit import gate for the two fields
that make paper rows comparable: model/prompt shape and batch or concurrency
policy. This prevents framework, MPK, VDCores, or controlled kernel rows from
entering the viewer with latency and throughput but without enough context to
interpret them.

## Evaluation Run

The guard was developed test-first. The fixture initially failed because the
validator accepted an MPK serving run that omitted
`batch_or_concurrency_policy`. After the validator and JSON contracts were
updated, the focused regression test passed and run-readiness data was
regenerated under
`tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-112a881d/`.

## Remaining Gaps

This is a metadata and evidence guard, not a new benchmark capture. MPK,
VDCores, vLLM, SGLang, and ThunderKittens serving-family runs still need
measured H200 artifacts before the LLM-serving paper-baseline claim can move
past `planned_no_results`.
