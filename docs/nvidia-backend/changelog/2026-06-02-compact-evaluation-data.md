# Compact Evaluation Data

## Code And Data Changed

- Moved benchmark-viewer payloads from
  `docs/nvidia-backend/benchmark-viewer/data/` to compact JSON files under
  `evaluations/nvidia/benchmark-viewer/data/`.
- Updated viewer, refresh, import, preflight, and validation scripts to use the
  evaluation data root.
- Replaced sharded JSON writer output with one JSON file per logical data set.
- Reduced `examples/cuda/manifest.json` and `examples/cuda/README.md` to the
  representative CUDA paths: host-schedule vector ops, persistent layered-cross
  graph, Qwen task-body generation, and Qwen decode-loop runner.

## Architecture Quality

Evaluation state is no longer stored directly inside `docs/`. The docs tree
keeps the HTML viewer and narrative reports; `evaluations/` owns data that is
expected to change as benchmark runs progress.

The examples catalog is now review-facing instead of probe-facing. Narrow Qwen
bring-up scripts remain available as support code, but they no longer crowd the
primary example manifest.

## Evaluation Run

- Regenerated derived NVIDIA review artifacts with:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
  ```

  The command passed and wrote compact files under
  `evaluations/nvidia/benchmark-viewer/data/`.
- The compact viewer data directory contains 20 JSON files instead of hundreds
  of sharded record files.

## Remaining Gaps

- `evaluations/nvidia/benchmark-viewer/data/results.json`
- `evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json`
- `examples/cuda/manifest.json`

This reorganizes review artifacts and examples; it does not provide the still
missing full-serving PTO, MPK, or VDCores paper-ready benchmark rows.
