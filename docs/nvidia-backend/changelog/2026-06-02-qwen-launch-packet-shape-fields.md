# 2026-06-02 Qwen Launch-Packet Shape Fields

## Code And Data Changed

- Added task-shape metadata normalization for CUDA persistent Qwen launch
  packets.
- Threaded workload-level shape defaults from submission plans into
  `CudaPersistentDagTask` records.
- Preserved future per-callable `task_shape_fields` and `scalar_fields` during
  resident-weight descriptor materialization.

## Architecture Quality

The Qwen persistent-device path now has an explicit contract for CUDA task ABI
fields such as `rows`, `cols`, `inner`, leading dimensions, batch strides, and
two scalar slots. Descriptor-local fields override workload defaults, which lets
future projection kernels carry model-specific matrix shapes without rewriting
the launch-packet builder.

The per-task `n` execution extent is derived by the runtime workspace instead
of workload token-count metadata. Follow-up shape-aware workspace work now uses
descriptor output shapes to size each activation buffer and to set each
non-final task's `n`.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py`.

## Remaining Gaps

This change only carries shape metadata to the persistent task ABI. Follow-up
generated source work uses those fields for projection and logits linear
bodies. Full-serving Qwen rows still require full attention, sampling, and
complete decode-loop execution.
