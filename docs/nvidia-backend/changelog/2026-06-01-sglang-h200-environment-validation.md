# 2026-06-01 SGLang H200 Environment Validation

## Code And Data Changed

- Captured the H200 SGLang editable install step from the pinned
  `tmp/baselines/sglang` checkout into the isolated environment.
- Captured the SGLang validation import window for the installed package and
  benchmark modules.
- Appended the distilled setup and validation attempt records to the benchmark
  viewer environment-attempt data. Raw command logs remain under `tmp/`.

## Architecture Quality

SGLang now has the same environment evidence shape as vLLM: dependency setup
is isolated in `tmp/cuda-backend/paper-baselines/envs/`, every step records
the exact command and log path, and validation imports run with
`PYTHONNOUSERSITE=1`. This makes serving benchmark claims depend on explicit
environment evidence instead of accidental user-site or project-venv state.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree synced from the current
branch. No upstream repository was edited or pushed.

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step03/`
- `tmp/cuda-backend/paper-baselines/environment-attempts/sglang-df219d33-h200-step04-09/`

Captured steps:

| Step | Kind | Status | Duration (s) |
| ---: | ---- | ------ | -----------: |
| 3 | install | pass | 147.943 |
| 4 | validation | pass | 9.34 |
| 5 | validation | pass | 0.047 |
| 6 | validation | pass | 1.832 |
| 7 | validation | pass | 9.809 |
| 8 | validation | pass | 12.511 |
| 9 | validation | pass | 12.859 |

Validated imports:

- `sglang`
- `orjson`
- `torchvision`
- `sglang.bench_serving`
- `sglang.bench_offline_throughput`
- `sglang.bench_one_batch`

The editable install placed SGLang and its CUDA serving dependencies in the
isolated environment, including Torch `2.11.0`, torchvision `0.26.0`, orjson
`3.11.9`, and FlashInfer `0.6.11.post1`.

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

- Run SGLang server launch and serving benchmark captures for the shared
  Qwen3-8B serving workload.
- Run SGLang offline throughput and one-batch benchmark captures.
- Import SGLang benchmark result records into the viewer once raw JSON exists.
