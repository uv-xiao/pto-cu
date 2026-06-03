# 2026-06-03 ThunderKittens LLM Policy Exception Guard

## Code And Data Changed

The paper-readiness audit now validates every `evidence_policy_exceptions`
entry before writing generated audit data. Each exception must include the
review-facing decision, rationale, review rule, status, scope, and concrete
evidence references.

`evaluations/nvidia/benchmark-viewer/data/paper_evaluation_matrix.json` now
records
`thunderkittens_llm_non_full_serving_policy_pending` under the
`llm_serving_paper_baselines` claim. The policy keeps the imported
ThunderKittens attention-tile rows classified as controlled attention-tile
proxy evidence until either full-serving Qwen/Qwen3-8B rows are imported or a
reviewed paper-table exception is accepted.

## Architecture Quality

Malformed policy exceptions are now rejected by the generator instead of being
silently copied into `paper_readiness_audit.json`. This keeps matrix policy,
generated viewer data, and review rules aligned before a claim can be promoted.

## Evaluation Run

RED first showed that a malformed LLM policy exception containing only an ID
and status was accepted by
`.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py`.

The focused regression now covers the missing `review_rule` failure path and
the real ThunderKittens LLM policy record:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_rejects_malformed_policy_exception \
  tests/ut/py/test_nvidia_review_artifacts.py::test_ultimate_goal_artifacts_define_paper_ready_cuda_path \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_schema_validator_passes \
  -q
```

## Remaining Gaps

ThunderKittens full-serving Qwen/Qwen3-8B rows are still missing. Until those
rows are imported, the LLM-serving claim remains non-paper-ready for the
ThunderKittens baseline contribution.
