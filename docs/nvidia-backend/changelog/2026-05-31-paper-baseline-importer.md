# 2026-05-31 Paper Baseline Importer

## Code And Data Changed

- Added `paper_baseline_viewer_export.py` under the CUDA backend evaluation
  skill.
- Added a benchmark-viewer workload entry for LLM serving decode.
- Added MPK, VDCores, vLLM, SGLang, and ThunderKittens as paper-baseline
  methods so imported paper-baseline rows can reference stable `method_id`
  values.
- Extended focused review tests and the NVIDIA review guard to require the
  paper-baseline importer, workload, methods, and changelog report.
- Documented the raw paper-baseline JSON import path in the evaluation skill,
  shared contracts, and paper-ready evaluation plan.

## Architecture Quality

Paper-baseline results now have the same import discipline as PTO CUDA
microbenchmark captures. Raw MPK, VDCores, vLLM, SGLang, and ThunderKittens
artifacts stay under `tmp/`; the importer normalizes them into
benchmark-viewer `result_records` with workload, method, hardware, statistic,
correctness, commit, and raw artifact fields.

This prevents future paper-baseline comparisons from becoming hand-edited
viewer rows. The mapping from a raw baseline run to a viewer method comes from
`paper_baseline_runs.json`, so the run contract, evaluation matrix, and result
table stay connected.

## Evaluation Run

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

.venv/bin/python -m py_compile \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py

git diff --check
```

The focused test fixture runs the importer on a vLLM serving raw JSON row and
checks the emitted `llm_serving_decode` / `vllm` result record. A second
fixture verifies boolean sample counts are rejected instead of being treated as
integer timing data.

## Remaining Gaps

- The importer defines the paper-baseline result path; it does not mean MPK,
  VDCores, vLLM, SGLang, or ThunderKittens have been built and benchmarked.
- Future evaluation slices must run the baseline commands, store raw artifacts
  under `tmp/cuda-backend/paper-baselines/`, import those rows, and update the
  viewer only with measured data.
