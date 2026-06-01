# 2026-06-01 vLLM H200 Serving Capture

## Code And Data Changed

- Added env-local `pandas`, `numexpr`, and `bottleneck` to the vLLM isolated
  environment plan so H200 model inspection does not import binary-incompatible
  system packages through `--system-site-packages`.
- Added `vllm.model_executor.models.qwen3` to the vLLM validation imports so
  the Qwen3 serving path is checked before server launch.
- Imported the H200 `Qwen/Qwen3-8B` batch-1 `vllm bench serve` capture into
  the benchmark viewer result data.
- Added H200 vLLM environment-attempt and execution-attempt records that point
  at the preserved raw artifacts under `tmp/cuda-backend/`.

## Architecture Quality

The vLLM environment plan now encodes the package-isolation issue discovered
while serving on H200 instead of leaving it as a one-off manual repair. The
validation surface also reaches the model module that failed during server
initialization, so future environment attempts can catch this class of failure
before a long-running serving command.

The viewer import stays scoped as a single successful batch-1 serving point.
It does not mark the full `vllm_serving_and_throughput` run as paper-ready.

## Evaluation Run

The H200 checkout was refreshed through the documented tree-sync fallback.
Repository Actions stayed disabled, and no upstream repository was edited or
pushed.

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-840a847f-h200-full/`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-840a847f-h200-validation/`
- `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-vdcores-qwen3-8b-batch1-840a847f-pandas/`

Serving result:

- model: `Qwen/Qwen3-8B`
- GPU: H200 on `bizhaoh200`
- request shape: input tokens `128`, output tokens `64`, batch/concurrency `1`
- completed requests: `1`
- failed requests: `0`
- mean TTFT: `64.89939196035266 ms`
- mean ITL: `5.8076567134805135 ms`
- output throughput: `148.25853760123334 tokens/s`
- request throughput: `2.316539650019271 req/s`

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result:

```text
benchmark viewer data validation passed
```

## Remaining Gaps

- Capture vLLM batch `2`, `4`, `8`, and `16` under the same Qwen3-8B policy.
- Repeat vLLM serving samples for variance and paper confidence intervals.
- Capture the matching MPK serving path and import it into the viewer.
- Compare the same serving workload against PTO persistent-device once that
  runner is available.
