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

## DeepSeek V4 Flash Artifact Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

This combines local config/tokenizer/index/shard readiness with the existing
weight-free vLLM DeepSeek V4 import and synthetic config probes. Missing local
artifacts or missing vLLM report structured skips by default; use
`--require-artifacts` or `--require-vllm` to make either condition fail the
command. It does not attempt model load, start a server, or run inference.

The remote H200 readiness slices are recorded in
`docs/in_progress/nvidia_backend/vllm_remote_install_probe.md` and
`docs/in_progress/nvidia_backend/deepseek_v4_flash_serving_readiness.md`.
They record remote H200 reachability and the current serving-readiness
boundary.
The follow-up environment/artifact gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`: the
remote vLLM import/config probes pass in `.venv-vllm-probe`, while the
artifact gates fail because the repo-relative artifact path contains
metadata/tokenizer files but not the indexed weight shards. The artifact
completion gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_artifact_complete.md`: the same
remote vLLM import/config probes pass, and the artifact/manifest gates now
find all 46 indexed shards at the repo-relative artifact path. These gates are
not model-load or serving evidence.

## DeepSeek V4 Flash vLLM Model-Load Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Run it only under an explicit GPU boundary, for example
`CUDA_VISIBLE_DEVICES=<two ids>` with a matching `--tensor-parallel-size 2`,
and an external timeout. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_model_load_probe.md`: vLLM loaded
all 46 shards and initialized an `LLMEngine` on two H200 GPUs at
`max_model_len=4096`. This is model-load and engine-initialization evidence,
not server health, inference correctness, 256K context, throughput, latency, or
production-readiness evidence.
