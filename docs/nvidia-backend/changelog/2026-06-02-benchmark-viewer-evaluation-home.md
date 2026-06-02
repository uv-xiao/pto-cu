# Benchmark Viewer Evaluation Home

## Code And Data Changed

- Moved the static benchmark-viewer UI from
  `docs/nvidia-backend/benchmark-viewer/` to
  `evaluations/nvidia/benchmark-viewer/viewer/`.
- Kept benchmark data under `evaluations/nvidia/benchmark-viewer/data/`.
- Updated the viewer config to load data with `../data/*.json` paths from its
  new evaluation-local home.
- Updated review guards, focused tests, current evaluation docs, and generated
  goal-progress evidence refs to the new viewer path.

## Architecture Quality

Evaluation artifacts now live under `evaluations/` instead of inside stable
NVIDIA backend design docs. This keeps `docs/nvidia-backend/` focused on
architecture, status, and changelog reports while preserving a directly
openable HTML viewer for human review.

This change does not add new checker-only functionality. It tightens the
project layout around the existing evaluation surface so future result imports
grow the evaluation area, not the design-doc tree.

## Evaluation Run

No new benchmark run was imported for this slice.

The updated viewer JavaScript passed syntax checks:

```bash
node --check evaluations/nvidia/benchmark-viewer/viewer/viewer.js
for f in evaluations/nvidia/benchmark-viewer/viewer/viewer/*.js; do
  node --check "$f"
done
```

The goal-progress JSON was regenerated with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py \
  --output evaluations/nvidia/benchmark-viewer/data/goal_progress.json
```

## Remaining Gaps

The viewer data remains under `evaluations/nvidia/benchmark-viewer/data/` and
should stay limited to review-meaningful JSON. Raw benchmark logs, large run
artifacts, and source checkouts should continue to stay under `tmp/`.
