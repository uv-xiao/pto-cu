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
- The synced remote checkout does not have `.venv-vllm-probe/bin/python`.
- System Python on the remote does not have importable `vllm`.
- The repo-relative remote artifact path
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash` is missing.
- The existing required vLLM/artifact probes return structured failures or
  skips, with exit code `2`, when run with `--require-vllm` and
  `--require-artifacts`.

## Serving Readiness State

The current remote state is blocked before serving can be tested:

```text
remote_h200_reachable: yes
remote_vllm_environment: missing
remote_repo_relative_artifacts: missing
weight_free_require_vllm_probes: failed/skipped because vLLM is missing
artifact_require_probe: failed because artifacts and vLLM are missing
serving_readiness: not established
```

This means the next reviewable step is still environment/artifact readiness on
the H200 checkout. It is too early to claim model-load readiness, server
readiness, or generated text correctness.

## Next Gate

The next PR-sized gate should:

1. Create or document a remote-local vLLM probe venv, such as
   `.venv-vllm-probe`, without committing dependencies or lockfiles.
2. Expose the already-available model artifacts through the repo-relative
   remote path `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash`, using
   symlinks under `tmp/` if the raw shards live elsewhere on the host.
3. Rerun:

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

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 vLLM engine initialization evidence.
- This is not vLLM server health evidence.
- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This did not copy, sync, download, or commit raw model artifacts.
