# 2026-05-31 PTO Serving-Equivalent Evidence

## Code And Data Changed

- Added a `llm_serving_decode` viewer result for PTO `persistent_device` on
  H200, using the existing tensor-core persistent DAG capture as a controlled
  `vdcores_offline_decode` attention-tile proxy.
- Added the matching `viewer_result` and raw-artifact evidence refs to the
  LLM serving paper-evaluation claim.
- Updated the serving workload and paper-readiness docs to state that PTO now
  has a controlled serving-policy-shaped row, while full end-to-end serving
  evidence remains absent.
- Regenerated `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json`.

## Architecture Quality

The new result is deliberately labeled
`pto_controlled_serving_equivalent`. It does not claim full LLM serving. The
row preserves the serving-policy metadata needed by reviewers: batch size,
prompt tokens, decode tokens, H200 hardware, persistent-device resource policy,
and the raw tmp artifact that produced the timing numbers.

## Evaluation Run

- Source artifact:
  `tmp/cuda-backend/graph-tensor-core-compact-current-working/combined-current-493ce832/`
- Imported row:
  `llm_serving_decode` / `pto_persistent_device` / `H200`
- Effect on paper readiness: the LLM serving claim drops from two missing
  evidence items to one. The remaining matrix gap is importing MPK, VDCores,
  vLLM, SGLang, and ThunderKittens-family raw outputs into viewer result
  records.

The focused TDD test first failed because the LLM serving matrix still listed
the PTO serving-equivalent gap. After adding the viewer result and matrix
evidence refs, the focused review-data test passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  -q
```

## Remaining Gaps

This slice does not run MPK, VDCores, vLLM, SGLang, or full ThunderKittens
serving-family captures. It also does not make PTO end-to-end LLM serving
available; the imported row is a controlled attention-tile proxy only.
