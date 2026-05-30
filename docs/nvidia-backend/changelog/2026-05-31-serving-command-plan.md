# 2026-05-31 Serving Command Plan

## Code And Data Changed

Added
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`.
The script reads `serving_workloads.json` and `paper_baseline_runs.json`, then
expands the MPK/VDCores serving policy contract into one command-plan row per
baseline, policy, and batch size.

The CUDA evaluation skill, review guard, and focused artifact tests now include
the command-plan generator. The in-progress evaluation plan, baseline survey,
and dispatch log now point to the generated `tmp/` artifact:
`tmp/cuda-backend/paper-baselines/serving-runs/plan-7cad653c.json`.

## Architecture Quality

The command-plan generator keeps policy selection, baseline command syntax, and
raw artifact paths in one checked path instead of leaving each long H200 run to
hand-build commands. It does not edit or patch upstream repositories; it only
materializes commands against the cloned sources and framework CLIs.

Each command row records the serving policy ID, paper-baseline run ID,
baseline ID, model tier, model, prompt tokens, decode tokens, batch size,
traffic mode, commands, and expected raw artifact paths. That gives future
baseline runners a stable handoff into `paper_baseline_viewer_export.py`.

## Evaluation Run

Generated the first primary-model command plan:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py \
    --output tmp/cuda-backend/paper-baselines/serving-runs/plan-7cad653c.json
```

The output JSON syntax was checked with:

```bash
.venv/bin/python -m json.tool \
  tmp/cuda-backend/paper-baselines/serving-runs/plan-7cad653c.json
```

The generated plan has 30 rows: MPK over one serving policy, VDCores over one
serving policy, and vLLM/SGLang over both serving policies, each expanded over
batch sizes 1, 2, 4, 8, and 16.

## Remaining Gaps

This is launch-plan evidence, not performance evidence. The MPK, VDCores,
vLLM, SGLang, and PTO serving-equivalent commands still need to run on H200,
write the raw artifacts named by the plan, and import normalized results into
the benchmark viewer.
