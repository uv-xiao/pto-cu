# 2026-06-02 Qwen Shape-Aware Workspace

## Code And Data Changed

- Threaded materialized resident Qwen task descriptors into activation
  workspace planning for both dry-run and single-context CUDA-live paths.
- Added per-activation-buffer element and byte counts derived from workload
  rows and descriptor-local output `cols`.
- Updated resource-backed launch packets so non-final tasks use their own
  activation output-buffer extent for `n`, while the logits task reports its
  logits extent and previous activation input extent.

## Architecture Quality

The activation workspace now follows the same descriptor shape contract as the
CUDA persistent DAG task ABI. Hidden-size buffers remain the fallback when a
descriptor has no output shape, but QKV, MLP gate/up, and other widened
intermediate outputs can now allocate and launch with model-shaped extents.

The launch preflight also reports `activation_buffer_element_counts`, giving
reviewers explicit evidence for the per-task `n` values used by the generated
host task packet.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_graph_materialization.py -q`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_single_context_session.py -q`.

## Remaining Gaps

This removes one full-model shape blocker for resource-backed Qwen serving.
Follow-up generated source work uses descriptor shapes for projection and
logits linear bodies and full-vocab argmax source. The branch still needs full
attention and complete decode-loop execution before PTO full-serving rows can
be imported as paper-ready evidence.
