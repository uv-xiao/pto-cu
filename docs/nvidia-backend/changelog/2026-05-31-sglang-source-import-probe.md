# 2026-05-31 SGLang Source Import Probe

## Code And Data Changed

Added a `python_import` readiness check to
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`. Unlike the
existing `python_module` check, it imports a selected module with an optional
source-relative `PYTHONPATH`, so SGLang benchmark entrypoints are checked from
`tmp/baselines/sglang/python`.

Updated `paper_serving_command_plan.py` so generated SGLang commands prepend
the pinned source checkout to `PYTHONPATH`. Refreshed the benchmark-viewer
probe data to point at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`, and
refreshed the serving command-plan reference to
`tmp/cuda-backend/paper-baselines/serving-runs/plan-43b927ed.json`.

## Architecture Quality

The evaluation contract now separates three states that were previously easy
to confuse: source checkout exists, Python can resolve a top-level package, and
the benchmark module can actually import from the pinned source tree. This
keeps the viewer from marking SGLang setup-ready until the callable benchmark
modules used by the paper plan can load.

The command generator also makes the source selection explicit in each SGLang
launch row, which prevents global site-packages from silently shadowing the
pinned checkout during remote evaluation.

## Evaluation Run

Regenerated the primary serving command plan:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py \
    --output tmp/cuda-backend/paper-baselines/serving-runs/plan-43b927ed.json
```

Reran paired A100/H200 readiness with remote tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py \
    --sync-remote-tree
```

The refreshed probe artifact is
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`. MPK,
VDCores, and ThunderKittens pass on both machines. vLLM remains partial on
both. SGLang remains partial because H200 is missing `orjson`, while local A100
SGLang source imports currently hit a torch/torchvision
operator-registration mismatch.

## Remaining Gaps

This is still readiness evidence, not a serving benchmark result. SGLang needs
its source runtime dependency stack installed on H200, and the local A100
torch/torchvision mismatch needs to be resolved before source-path SGLang
serving and offline-throughput captures can be imported into the viewer.
