# 2026-06-01 Serving Work Queue Targets

## Code And Data Changed

- Added structured `missing_evidence_details` to the LLM-serving paper
  evaluation matrix for the remaining Qwen3-8B serving gaps.
- Propagated those details through the paper-readiness audit and flattened
  work queue, including method, baseline, serving workload, and shape target.
- Updated the benchmark viewer so the paper work queue shows serving workload
  and shape columns, and audit next actions include the same target context.
- Extended the benchmark-viewer data guard and unit tests to reject stale or
  invalid serving-workload targets.

## Architecture Quality

The paper-readiness queue now exposes actionable units instead of one broad
LLM-serving blocker. Reviewers can see that the next missing rows are PTO
full-serving Qwen3-8B, MPK persistent Qwen3-8B, VDCores full-serving Qwen3-8B,
SGLang MPK-policy Qwen3-8B, and ThunderKittens-family full-serving Qwen3-8B.
This makes later H200 imports auditable against explicit method and serving
policy targets.

## Evaluation Run

This slice changed review data and validation code only; it did not run new
GPU benchmarks. Validation commands:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
    .agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py \
    .agents/checks/validate_benchmark_viewer_data.py
```

```bash
jq empty docs/nvidia-backend/benchmark-viewer/data/*.json
```

## Remaining Gaps

- The work queue is more precise, but the new target rows are still missing.
- Paper readiness still requires measured H200 artifacts for the explicit
  Qwen3-8B serving targets before promotion.
