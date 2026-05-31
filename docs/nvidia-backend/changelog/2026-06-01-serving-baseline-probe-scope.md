# 2026-06-01 Serving Baseline Probe Scope

## Code And Data Changed

- Extended `paper_baseline_probe.py` so Python module and import checks can run
  with `PYTHONNOUSERSITE=1` and an explicit source `PYTHONPATH`.
- Updated vLLM probe coverage to distinguish pinned source import readiness
  from installed runtime readiness.
- Updated SGLang probe coverage to check isolated `orjson` and `torchvision`
  dependency availability before benchmark module imports.
- Refreshed paired A100/H200 probe artifacts and regenerated run-readiness,
  audit, work-queue, and goal-progress viewer data.

## Architecture Quality

The serving baseline probes now separate three states that were previously
collapsed into one partial status:

- source entrypoints exist and compile;
- pinned source packages can be imported with an explicit source path;
- runtime dependencies needed by server and benchmark entrypoints are still
  missing from the evaluation environments.

This makes the next evaluation action concrete without installing large vLLM or
SGLang dependency stacks into the shared project venv. SGLang checks also
isolate user-site packages so local `orjson` or incompatible `torchvision`
installs cannot hide the true evaluation-host state.

## Evaluation Run

Paired probe command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py \
  --sync-remote-tree --local-python .venv/bin/python
```

Artifacts:

- `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-86ea3913/a100-probe.json`
- `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-86ea3913/h200-probe.json`
- `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-86ea3913/paired-probe-summary.json`

Observed vLLM status on A100 and H200:

- source `vllm` import passed with `PYTHONPATH=tmp/baselines/vllm`;
- `vllm.entrypoints.cli.main` import passed;
- installed `vllm` module check still failed;
- server and engine argument imports failed on missing runtime dependencies
  such as `uvloop`, `pydantic`, and `cbor2`.

Observed SGLang status:

- source benchmark files exist and compile;
- isolated `orjson` and `torchvision` checks failed;
- benchmark module imports failed first on missing `orjson` under
  `PYTHONNOUSERSITE=1`.

## Remaining Gaps

- Install vLLM runtime dependencies in an isolated evaluation environment
  before running `vllm serve`, `vllm bench serve`, and
  `vllm bench throughput`.
- Install SGLang runtime dependencies in an isolated evaluation environment
  before running `sglang.launch_server`, `sglang.bench_serving`,
  `sglang.bench_offline_throughput`, and `sglang.bench_one_batch`.
- Do not install these large framework stacks into the shared project venv
  unless that is made an explicit evaluation-environment decision.
