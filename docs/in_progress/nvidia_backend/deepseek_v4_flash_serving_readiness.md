# DeepSeek V4 Flash Serving Readiness

This note records the current remote H200 readiness state for
`deepseek-ai/DeepSeek-V4-Flash` after the local artifact and weight-free vLLM
probe gates. It is a readiness gap record, not serving evidence.

## Current Evidence

Already merged inputs:

- The vLLM DeepSeek V4 import/config probes exist and are weight-free.
- The local DeepSeek-V4-Flash weight manifest gate found all 46 indexed shards
  present in a gitignored artifact directory.
- The local artifact readiness probe can compose artifact inspection with the
  existing vLLM import/config probes.

Fresh remote H200 evidence from this slice:

- The remote H200 runner is reachable.
- The remote reports 8 x NVIDIA H200 NVL GPUs, compute capability 9.0,
  driver 580.126.20, 143771 MiB per GPU, and CUDA 12.8 `nvcc`.
- A follow-up remote-local `.venv-vllm-probe` was created and vLLM was
  installed only into that venv: vLLM 0.23.0, Torch 2.11.0, transformers
  5.12.1, flashinfer-python 0.6.12, and Triton 3.6.0.
- The vLLM probe venv now imports `vllm` and passes `pip check` after
  venv-local compatibility fixes for system-site package shadowing, including
  `scipy`, `blosc2`, `pandas`, and `bottleneck`.
- The weight-free DeepSeek V4 vLLM import probe passes with
  `--require-vllm`.
- The weight-free DeepSeek V4 vLLM config probe passes with
  `--require-vllm --max-position-embeddings 262144`.
- The repo-relative remote artifact path
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash` exposes metadata and
  tokenizer files, but no indexed weight shards.
- The artifact probe and weight manifest gate fail with exit code `2` because
  artifacts are incomplete: 46 indexed shards, 0 present shards, and
  159609485896 indexed bytes.

Detailed follow-up evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`.

## Serving Readiness State

The current remote state is blocked before serving can be tested:

```text
remote_h200_reachable: yes
remote_vllm_environment: ready for weight-free import/config probes
remote_repo_relative_artifacts: metadata/tokenizer only
weight_free_require_vllm_probes: passed
artifact_require_probe: failed because indexed shards are missing
weight_manifest_gate: incomplete because indexed shards are missing
serving_readiness: not established
```

This means the next reviewable step is remote artifact completion on the H200
checkout. It is too early to claim model-load readiness, server readiness, or
generated text correctness.

## Next Gate

The next PR-sized gate should:

1. Expose a complete model artifact directory through the repo-relative remote
   path `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash`, using symlinks
   under `tmp/` if the raw shards live elsewhere on the host.
2. Rerun:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_import_probe.py --require-vllm
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_config_probe.py \
  --require-vllm --max-position-embeddings 262144
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm
```

That gate should remain weight-free unless the dispatcher explicitly assigns a
model-load or serving slice.

Also rerun the weight manifest gate against the same path:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-complete
```

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 vLLM engine initialization evidence.
- This is not vLLM server health evidence.
- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This did not copy, sync, download, or commit raw model artifacts.
