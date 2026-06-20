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

Fresh remote H200 server health evidence from this slice:

- A repo-owned server-health probe script now launches `vllm serve`, binds
  only to `127.0.0.1`, polls readiness endpoints, captures structured JSON,
  terminates the server, and checks for remaining process-group PIDs.
- The installed remote vLLM 0.23.0 server CLI was inspected before choosing
  flags. The probe uses `.venv-vllm-probe/bin/vllm serve [model_tag]` with
  inspected OpenAI server and model-load flags.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. They still had enough free memory for the 0.78
  utilization boundary before the run.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 50-minute outer timeout, and a
  2700-second readiness timeout.
- The server started on `http://127.0.0.1:28123`, returned HTTP 200 from
  `/health`, and returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`.
- The probe exited 0 after 114.348 seconds with `generation_attempted=false`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  snapshot showed the selected GPUs back at their pre-run memory baseline.
- The run established local-only server startup and health/model-list
  readiness only; it did not run generation.

Detailed server-health evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_server_health_probe.md`.

Fresh remote H200 inference-smoke evidence from this slice:

- A repo-owned inference-smoke probe script now reuses the server-health
  lifecycle, checks `/health` and `/v1/models`, sends one bounded inference
  request, records response shape, terminates the server, and checks for
  remaining process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 55-minute outer timeout, a
  2700-second readiness timeout, and a 180-second request timeout.
- The server started on `http://127.0.0.1:28124`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted one `/v1/completions` request with prompt `Hello`, `max_tokens=1`,
  `temperature=0.0`, and `stream=false`.
- The one inference request returned HTTP 200 with one response choice. The
  probe recorded response shape and request limits only; it did not compare
  generated text against an expected answer.
- The probe exited 0 after 213.558 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  snapshot showed the selected GPUs back at their pre-run memory baseline.
- The run establishes one-token inference-smoke evidence only; it is not
  generated-text correctness evidence.

Detailed inference-smoke evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_inference_smoke_probe.md`.

## Serving Readiness State

The current remote state is complete through bounded two-H200 vLLM
server-startup, health/model-list readiness, and one-token inference smoke:

```text
remote_h200_reachable: yes
remote_vllm_environment: ready for weight-free import/config probes
remote_repo_relative_artifacts: complete for indexed shard presence
weight_free_require_vllm_probes: passed
artifact_require_probe: passed
weight_manifest_gate: complete
two_h200_vllm_model_load: passed under recorded 4096-token boundary
local_only_vllm_server_health: passed under recorded 4096-token boundary
one_token_inference_smoke: passed under recorded 4096-token boundary
serving_readiness: one bounded local-only request; no correctness claims
```

This means the remote H200 environment has passed a local-only vLLM server
startup and one bounded local-only inference request under the recorded
two-H200 boundary. It is still too early to claim generated text correctness,
long-context behavior, latency, throughput, production readiness, or
simpler-nv/vLLM kernel integration.

## Next Gate

The next PR-sized gate can decide whether to reduce first-request JIT/TileLang
compilation surprises with an explicit warmup shape, or move to a separate
serving-semantics check. That later gate needs its own command, resource plan,
expected failure mode, and non-claims, and it should stay separate from
correctness, throughput, latency, long-context, and production-readiness
claims.

## Non-Claims

- This is not prompt, tokenizer semantic, correct-text, long-context,
  throughput, latency, or production readiness evidence.
- This is not generated-text correctness evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.
