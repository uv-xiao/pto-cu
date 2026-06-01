# 2026-06-01 SGLang H200 Probe Narrowing

## Code And Data Changed

- Updated `paper_baseline_probes.json` so the SGLang probe now separates the
  already validated H200 environment from the still-missing A100 validation.
- Regenerated run-readiness, paper-readiness audit, and work-queue data from
  the updated probe state.
- Updated review-artifact tests so the expected SGLang blocker is now only the
  A100 validation gap.

## Architecture Quality

The SGLang readiness surface now matches the evidence hierarchy. The H200
probe points at the isolated environment-validation artifact, while A100 keeps
the older missing-dependency blockers until it is validated in a matching
isolated environment. This avoids presenting an already resolved H200 import
gap as an active paper-readiness blocker.

## Evaluation Run

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step04-09/`
- `tmp/cuda-backend/paper-baselines/probes/sglang-h200-env-df219d33/`

The H200 validation artifact shows passing imports for:

- `sglang`
- `orjson`
- `torchvision`
- `sglang.bench_serving`
- `sglang.bench_offline_throughput`
- `sglang.bench_one_batch`

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

- Materialize and validate the SGLang isolated environment on A100.
- Keep the SGLang probe partial until both A100 and H200 machine statuses pass.
