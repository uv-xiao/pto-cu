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

Fresh remote H200 model-load evidence from this slice:

- A repo-owned model-load probe script now instantiates `vllm.LLM` with
  structured output and no serving or generation path.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were 1 and 7, exposed as exactly two visible
  devices with `CUDA_VISIBLE_DEVICES=1,7` and `tensor_parallel_size=2`.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, and a 45-minute timeout.
- The first attempt failed usefully with `kv_cache_dtype=auto` and
  `gpu_memory_utilization=0.82`: vLLM required fp8 KV cache for the DeepSeek
  V4 FlashMLA fp8 layout, and the second selected GPU had less free memory
  than the requested utilization boundary.
- The retry passed: all 46 safetensors shards were loaded, `vllm.LLM` returned
  an `LLMEngine`, and the probe exited 0 after 605.411 seconds.
- The passing run loaded weights and completed vLLM engine initialization only;
  it did not start a server or run inference.

Detailed model-load evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_model_load_probe.md`.

## Serving Readiness State

The current remote state is complete through bounded two-H200 vLLM model load
and engine initialization:

```text
remote_h200_reachable: yes
remote_vllm_environment: ready for weight-free import/config probes
remote_repo_relative_artifacts: complete for indexed shard presence
weight_free_require_vllm_probes: passed
artifact_require_probe: passed
weight_manifest_gate: complete
two_h200_vllm_model_load: passed under recorded 4096-token boundary
serving_readiness: not established
```

This means the next reviewable step is an explicit server startup and health
probe with its own resource boundary. It is still too early to claim server
readiness, generated text correctness, long-context behavior, latency,
throughput, or production readiness.

## Next Gate

The next PR-sized gate should run an explicit vLLM server startup and health
probe on the remote H200 checkout, or a separately justified one-token smoke if
server readiness requires generation. That later gate needs its own command,
resource plan, expected failure mode, and non-claims before it should make any
serving statement.

## Non-Claims

- This is not vLLM server health evidence.
- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.
