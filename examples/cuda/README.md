# CUDA Examples

These examples preserve review-facing CUDA backend metadata and provide small
skip-safe probes for current work. Historical benchmark rows do not run fresh
CUDA hardware checks; their A100/H200 measurements remain the `743709f3`
capture documented under `docs/nvidia-backend/history/`.

## Host-Schedule Vector Ops

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/host_schedule_vector_ops.py \
  --describe --op add --n 1024 --arch compute_80
```

Use `--op` to select the evaluated host-schedule ABI shape:
`add`, `mul`, `scale`, `square`, `axpy`, `affine`, `triad`, `quad`,
`generic_args`, or `generic_args4`.

## Persistent Layered-Cross Graph

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_layered_cross.py \
  --describe --n 1024 --arch compute_80 --scheduler-blocks 3
```

This describes the same `graph_descriptor_layered_cross` shape that feeds the
current `743709f3` benchmark gate.

## Persistent MoE Dispatch/Combine

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --output-json tmp/persistent-moe-dispatch-combine-local.json
```

This emits structured JSON for `graph_descriptor_moe_dispatch_combine`: four
expert transform tasks, one weighted combine task, and device-side fan-in
before the combine. Without CUDA tooling or a visible NVIDIA GPU it reports a
skip; with `--require-cuda`, the same skip returns a non-zero exit status.

## DeepSeek V4 Flash Weight Manifest

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --require-complete
```

This checks local gitignored shard presence against
`model.safetensors.index.json`. The completed local artifact evidence is
recorded in
`docs/in_progress/nvidia_backend/deepseek_v4_flash_weight_manifest_complete.md`.
It is not model-load or serving evidence.
