# 2026-06-02 Qwen Kernel Source Map

## Code And Data Changed

- Added `qwen_kernel_source_map` to the generated Qwen task-body manifest.
- Recorded FlashInfer, vLLM, and SGLang sparse source snapshots under
  `tmp/sources/kernel-references/` for local review.
- Added a concise reviewer doc at
  `docs/nvidia-backend/kernel-sources/qwen-source-map.md`.

## Architecture Quality

The map ties each incomplete Qwen callable area to concrete upstream CUDA
kernel families without vendoring external source or modifying upstream repos.
This makes the next implementation steps reviewable: PTO must move from
diagnostic formulas to model-shape gate/up buffers, QK norm/RoPE, slot-mapped
KV writes, decode attention, and tiled logits projection.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_task_body_math.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_cuda_examples.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py`.

## Remaining Gaps

This is source provenance, not serving correctness. Full-serving rows still
require numerically complete kernels and live latency/throughput imports.
