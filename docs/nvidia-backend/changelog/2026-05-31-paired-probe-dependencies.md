# 2026-05-31 Paired Probe Dependencies

## Code And Data Changed

Tightened the paper-baseline readiness probe contract for MPK and VDCores by
requiring an explicit `transformers` module check. Both selected entrypoints
import Transformers APIs for tokenizer, config, and model loading, so a probe
that only checked `torch` was too weak.

Updated `paper_baseline_probes.json` to point its latest paired evidence at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-57de1a6b/`, and
updated the evaluation plan plus dispatch log with the refreshed paired probe
state.

## Architecture Quality

The readiness contract now checks the model-loading stack that the baseline
entrypoints actually import. This keeps the viewer from marking MPK or VDCores
setup-ready when their Python model dependencies are incomplete.

The change stays within repo-owned contracts and environment setup. No
baseline source checkout or upstream repository was modified.

## Evaluation Run

Installed the missing H200 project-venv dependency:

```bash
ssh bizhaoh200 'cd /data/shibizhao/pto-cu && .venv/bin/python -m pip install transformers'
```

Then reran the paired probe with tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py \
    --sync-remote-tree
```

The refreshed artifact is
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-57de1a6b/`. A100
reports MPK, VDCores, SGLang, and ThunderKittens as pass and vLLM as partial.
H200 reports MPK, VDCores, and ThunderKittens as pass and vLLM/SGLang as
partial.

## Remaining Gaps

The paired probe now has stronger dependency evidence, but it is still setup
evidence. vLLM and SGLang remain partial on H200 until their packages are
installed or built in the project venv, and no serving performance rows have
been captured from the generated command plan yet.
