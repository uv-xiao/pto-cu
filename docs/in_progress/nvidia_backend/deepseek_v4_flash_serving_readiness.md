# DeepSeek V4 Flash Serving Readiness

This note records the current remote H200 readiness state for
`deepseek-ai/DeepSeek-V4-Flash` after the local artifact and weight-free vLLM
probe gates. It is a pre-load readiness record, not serving evidence.

## Current Evidence

Already merged inputs:

- The vLLM DeepSeek V4 import/config probes exist and are weight-free.
- The local DeepSeek-V4-Flash weight manifest gate found all 46 indexed shards
  present in a gitignored artifact directory.
- The local artifact readiness probe can compose artifact inspection with the
  existing vLLM import/config probes.

Merged remote H200 environment evidence:

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
- A prior remote artifact check showed the repo-relative artifact path exposed
  metadata/tokenizer files but no indexed weight shards.

Detailed follow-up evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`.

Fresh remote H200 artifact evidence from this slice:

- The complete `deepseek-ai/DeepSeek-V4-Flash` artifact directory was copied
  into the remote checkout's ignored repo-relative path:
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash`.
- The source tree was refreshed to commit
  `cf8fabaae2f25fb58b04d74f5c57517a9dab4ea3`; the existing
  `.venv-vllm-probe` was reused without repair.
- The weight-free DeepSeek V4 vLLM import probe passed with
  `--require-vllm`.
- The weight-free DeepSeek V4 vLLM config probe passed with
  `--require-vllm --max-position-embeddings 262144`.
- The composed artifact/vLLM probe passed with `--require-artifacts
  --require-vllm`: 46 indexed shards, 46 present shards, 0 missing shards,
  and 159617149040 present bytes.
- The standalone weight manifest gate passed with `--require-complete`:
  46 indexed shards, 46 present shards, 0 missing shards, and
  159617149040 present bytes.

Detailed artifact-complete evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_artifact_complete.md`.

## Serving Readiness State

The current remote state is complete through the weight-free pre-load artifact
gate:

```text
remote_h200_reachable: yes
remote_vllm_environment: ready for weight-free import/config probes
remote_repo_relative_artifacts: complete for indexed shard presence
weight_free_require_vllm_probes: passed
artifact_require_probe: passed
weight_manifest_gate: complete
serving_readiness: not established
```

This means the next reviewable step is an explicit model-load or serving probe
with its own resource boundary. It is still too early to claim vLLM engine
initialization, server readiness, generated text correctness, long-context
behavior, latency, throughput, or production readiness.

## Next Gate

The next PR-sized gate should run an explicit model-load or serving readiness
probe on the remote H200 checkout. That later gate needs its own command,
resource plan, expected failure mode, and non-claims before it should make any
serving statement.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 vLLM engine initialization evidence.
- This is not vLLM server health evidence.
- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.
