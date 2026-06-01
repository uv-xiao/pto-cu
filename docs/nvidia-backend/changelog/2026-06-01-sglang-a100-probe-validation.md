# 2026-06-01 SGLang A100 Probe Validation

## Code And Data Changed

- Materialized the isolated A100 SGLang environment under
  `tmp/cuda-backend/paper-baselines/envs/sglang-7ed53d15`.
- Captured A100 SGLang entrypoint imports and converted the logs into
  structured validation evidence under `tmp/`.
- Added a paired SGLang probe root that marks both A100 and H200 machine
  statuses as passing.
- Regenerated run-readiness, paper-readiness audit, work-queue, and goal
  progress data from the updated probe state.
- Updated review-artifact tests so serving-framework probes are no longer
  active paper-readiness blockers.

## Architecture Quality

SGLang now matches the vLLM serving-framework readiness shape: both machine
classes have isolated environment validation, per-module import logs, and
machine-status probe records. This removes dependency availability from the
paper-readiness blocker list while still keeping benchmark-result promotion
tied to raw serving artifacts and the shared workload policy.

## Evaluation Run

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-7ed53d15-a100-install/`
- `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-7ed53d15-a100-validation/`
- `tmp/cuda-backend/paper-baselines/probes/sglang-a100-h200-env-7ed53d15/`

Validated A100 imports:

- `sglang`
- `orjson`
- `torchvision`
- `sglang.bench_serving`
- `sglang.bench_offline_throughput`
- `sglang.bench_one_batch`

Verification commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

Result: passed.

## Remaining Gaps

- Keep SGLang benchmark-result promotion tied to imported raw serving artifacts
  and the shared workload policy.
- Continue reducing the remaining non-probe paper-readiness blockers in
  `paper_readiness_work_queue.json`.
