# 2026-05-31 Probe Status Updater

## Code And Data Changed

Added `.agents/skills/cuda-backend-eval/scripts/paper_probe_status_update.py`.
The script reads a paired A100/H200 probe artifact directory and regenerates
the committed `paper_baseline_probes.json` status fields:

- `latest_artifact_root`;
- aggregate `latest_status`;
- per-machine `latest_machine_status` for A100 and H200.

Added a focused test that builds a synthetic paired probe artifact and verifies
the script materializes the expected aggregate and per-machine status.

## Architecture Quality

This removes the manual JSON-edit step between raw readiness artifacts and the
benchmark viewer. Future paired probe refreshes can update the committed
viewer data mechanically, while the existing validator proves the committed
summary still matches the raw JSON.

## Evaluation Run

The script is intended to run after
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_probe_status_update.py \
    --paired-artifact-root tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-$(git rev-parse --short HEAD) \
    --output docs/nvidia-backend/benchmark-viewer/data/paper_baseline_probes.json
```

## Remaining Gaps

The updater keeps readiness summaries synchronized with raw probe artifacts,
but it does not install missing baseline dependencies or run performance
benchmarks. MPK, VDCores, vLLM, SGLang, and full ThunderKittens sweeps still
need captured raw results before paper rows can be promoted.
