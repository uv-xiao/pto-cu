# 2026-06-01 vLLM A100 Probe Validation

## Code And Data Changed

- Captured A100 vLLM entrypoint imports in the isolated
  `vllm-27fa5aa3` environment.
- Added a paired vLLM probe root that marks both A100 and H200 machine
  statuses as passing.
- Regenerated run-readiness, paper-readiness audit, work-queue, and goal
  progress data from the updated probe state.
- Updated review-artifact tests so vLLM is no longer an active probe blocker.

## Architecture Quality

The vLLM probe now uses the same evidence shape on both machine classes:
isolated environment imports under `tmp/`, per-module logs, and a probe JSON
that is referenced by committed viewer data. This keeps serving-framework
readiness separate from benchmark-result promotion while removing a stale
A100-only validation gap.

## Evaluation Run

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-27fa5aa3-a100-entrypoint-validation/`
- `tmp/cuda-backend/paper-baselines/probes/vllm-a100-h200-env-27fa5aa3/`

Validated A100 imports:

- `vllm`
- `vllm.entrypoints.cli.main`
- `vllm.entrypoints.openai.api_server`
- `vllm.engine.arg_utils`
- `vllm.model_executor.models.qwen3`

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

- Keep vLLM benchmark-result promotion tied to the imported raw serving
  artifacts and shared workload policy.
- Materialize the matching A100 SGLang environment before removing the final
  serving-framework probe blocker.
