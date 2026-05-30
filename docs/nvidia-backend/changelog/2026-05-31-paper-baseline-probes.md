# 2026-05-31 Paper Baseline Probes

## Code And Data Changed

- Added `paper_baseline_probe.py` under the CUDA backend evaluation skill.
- Added `paper_baseline_probes.json` to define safe readiness checks for MPK,
  VDCores, vLLM, SGLang, and ThunderKittens.
- Updated the benchmark viewer to render readiness probes next to each paper
  baseline.
- Extended the viewer-data validator, NVIDIA review guard, and focused review
  tests to require the probe data and script.
- Documented the probe workflow in the CUDA evaluation skill, shared
  contracts, and paper-ready evaluation plan.

## Architecture Quality

The paper-baseline workflow now has an intermediate evidence layer between
source survey and full benchmark results. Probe records are intentionally
safe: they check pinned source commits, selected source entrypoints, Python
syntax, required modules, visible GPUs, and CUDA toolkit availability without
starting model-serving jobs or modifying upstream repositories.

This makes baseline readiness human-reviewable in the same HTML viewer as
benchmark runs, while keeping raw probe outputs under `tmp/`.

## Evaluation Run

The local A100 probe was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py \
    --output tmp/cuda-backend/paper-baselines/probes/local-a100-source-entrypoints/probe.json \
    --artifact-root tmp/cuda-backend/paper-baselines/probes/local-a100-source-entrypoints/
```

The probe saw seven A100 GPUs and CUDA 12.8 `nvcc`. MPK, VDCores, SGLang, and
ThunderKittens passed their selected source-entrypoint probes. vLLM was
partial because source entrypoints parsed but the `vllm` Python module is not
installed in the current environment.

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

node --check docs/nvidia-backend/benchmark-viewer/viewer.js

git diff --check
```

## Remaining Gaps

- Probe pass is not benchmark completion. MPK, VDCores, vLLM, SGLang, and
  ThunderKittens still need build/install runs, model selection, benchmark
  execution, raw result capture, and viewer import.
- The current probe is local A100 evidence. H200 probe and full baseline runs
  still need the remote refresh or SSH tree-sync fallback path.
