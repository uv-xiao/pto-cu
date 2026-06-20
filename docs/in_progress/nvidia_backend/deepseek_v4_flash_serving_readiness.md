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

Fresh remote H200 response-contract evidence from this slice:

- A repo-owned response-contract probe script now reuses the server-health
  lifecycle, checks `/health` and `/v1/models`, sends one bounded
  `/v1/completions` request, validates structural response invariants,
  terminates the server, and checks for remaining process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 60-minute outer timeout, a
  2700-second readiness timeout, and a 180-second request timeout.
- The server started on `http://127.0.0.1:28125`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted one `/v1/completions` request with prompt `Hello`, `max_tokens=4`,
  `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, and `stream=false`.
- The completion request returned HTTP 200 with exactly one response choice,
  response `model`, choice `text` and `finish_reason`, and usage counts:
  `prompt_tokens=1`, `completion_tokens=4`, and `total_tokens=5`.
- `usage.completion_tokens` was within the request `max_tokens` bound, and
  `usage.total_tokens >= usage.prompt_tokens`.
- The response choice shape included a `token_ids` key, but the value was not
  a list in this response, so the token-id length check was recorded as
  `not_present` rather than asserting a count from a null field.
- The probe exited 0 after 109.204 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  snapshot showed the selected GPUs back at their pre-run memory baseline.
- The run establishes local-only response-contract evidence only; it is not
  generated-text correctness evidence.

Detailed response-contract evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_response_contract_probe.md`.

Fresh remote H200 warmup-shape evidence from this slice:

- A repo-owned warmup-shape probe script now reuses the response-contract
  lifecycle, checks `/health` and `/v1/models`, sends one labeled warmup
  `/v1/completions` request, sends one follow-up same-shape
  `/v1/completions` request, validates the existing structural response
  contract for both responses, inspects selected Triton JIT warning strings by
  server-log request windows, terminates the server, and checks for remaining
  process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, a 180-second request timeout, and a 2-second
  server-log settle interval after each completion request.
- The server started on `http://127.0.0.1:28126`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted two same-shape `/v1/completions` requests with prompt `Hello`,
  `max_tokens=4`, `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, and
  `stream=false`.
- Both completion requests returned HTTP 200 with exactly one response choice
  and structurally valid response-contract fields. Both responses reported
  `usage.prompt_tokens=1`, `usage.completion_tokens=4`, and
  `usage.total_tokens=5`.
- The selected warning pattern
  `Triton kernel JIT compilation during inference` appeared 8 times in the
  warmup request log window and 0 times in the follow-up same-shape request
  log window. This is request-window observation for this exact command and
  selected warning string, not a broad claim that warmup eliminates all JIT
  behavior.
- The probe exited 0 after 115.046 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes local-only warmup-shape observation evidence only; it is
  not generated-text correctness, latency, throughput, long-context, or
  production-readiness evidence.

Detailed warmup-shape evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_warmup_shape_probe.md`.

Fresh remote H200 request-shape variation evidence from this slice:

- A repo-owned request-shape variation probe script now reuses the
  warmup-shape lifecycle, checks `/health` and `/v1/models`, sends one known
  warmup `/v1/completions` request, sends one same-shape follow-up request to
  preserve the prior baseline inside the run, sends one separate bounded
  variation `/v1/completions` request with a different prompt/token shape,
  validates the structural response contract for all three responses, inspects
  selected Triton JIT warning strings by server-log request windows,
  terminates the server, and checks for remaining process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, a 180-second request timeout, and a 2-second
  server-log settle interval after each completion request.
- The server started on `http://127.0.0.1:28127`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`,
  accepted the known warmup and same-shape follow-up payloads with prompt
  `Hello` and `max_tokens=4`, then accepted a variation payload with prompt
  `Hello. Provide one short sentence about bounded CUDA serving probes.` and
  `max_tokens=8`.
- All three completion requests returned HTTP 200 with exactly one response
  choice and structurally valid response-contract fields. The variation
  response reported `usage.prompt_tokens=13`, `usage.completion_tokens=8`,
  and `usage.total_tokens=21`.
- The selected warning pattern
  `Triton kernel JIT compilation during inference` appeared 8 times in the
  warmup request log window, 0 times in the follow-up same-shape request log
  window, and 3 times in the variation request log window. The variation
  warning lines named `_build_prefill_chunk_metadata_kernel`,
  `_compute_prefill_metadata_kernel`, and `_combine_topk_swa_indices_kernel`.
- The probe exited 0 after 163.042 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes local-only request-shape variation observation evidence
  only; it is not generated-text correctness, tokenizer semantic correctness,
  prompt correctness, latency, throughput, long-context, production-readiness,
  broad warmup-eliminates-JIT, or simpler-nv/vLLM integration evidence.

Detailed request-shape variation evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_request_shape_variation_probe.md`.

Fresh remote H200 serving-semantics evidence from this slice:

- A repo-owned serving-semantics probe script now reuses the warmup-shape
  lifecycle and response-contract validation, checks `/health` and
  `/v1/models`, sends two identical bounded deterministic
  `/v1/completions` requests, compares selected API-boundary observations,
  inspects selected Triton JIT warning strings by server-log request windows,
  terminates the server, and checks for remaining process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, a 180-second request timeout, and a 2-second
  server-log settle interval after each completion request.
- The server started on `http://127.0.0.1:28128`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted two identical deterministic completion payloads with prompt
  `Hello. Keep this deterministic serving probe bounded.`, `max_tokens=8`,
  `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, and `stream=false`.
- Both completion requests returned HTTP 200 with exactly one response choice
  and structurally valid response-contract fields. Both responses reported
  `usage.prompt_tokens=9`, `usage.completion_tokens=8`, and
  `usage.total_tokens=17`.
- The deterministic-repeat serving-semantics comparison passed for completion
  text digest, text length, `finish_reason`, and usage accounting. The probe
  did not record the raw generated text and did not compare it against an
  expected answer.
- The selected warning pattern
  `Triton kernel JIT compilation during inference` appeared 11 times in the
  first request log window and 0 times in the repeat deterministic request
  log window.
- The probe exited 0 after 125.057 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes a local-only deterministic serving-semantics
  observation for this exact bounded request pair; it is not generated-text
  correctness, tokenizer semantic correctness, prompt correctness, latency,
  throughput, long-context, broad determinism, production-readiness, or
  simpler-nv/vLLM integration evidence.

Detailed serving-semantics evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_serving_semantics_probe.md`.

Fresh remote H200 logprobs-contract evidence from this slice:

- The repo-owned response-contract probe script now has a
  `--logprobs-contract` mode that reuses the existing server lifecycle,
  health/model-list readiness checks, response-contract validation, cleanup,
  and local-only server boundary.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, and a 180-second request timeout.
- The server started on `http://127.0.0.1:28129`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted one completion payload with `prompt=Hello`, `max_tokens=4`,
  `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, `stream=false`,
  `logprobs=1`, and `prompt_logprobs=1`.
- The completion request returned HTTP 200 with exactly one response choice
  and structurally valid response-contract fields. The response reported
  `usage.prompt_tokens=1`, `usage.completion_tokens=4`, and
  `usage.total_tokens=5`.
- The explicit logprobs response-shape checks passed:
  `choice.logprobs` exposed list-valued completion logprob fields with
  lengths matching `usage.completion_tokens`, and `choice.prompt_logprobs`
  was list-valued and bounded by `usage.prompt_tokens`.
- The probe exited 0 after 110.636 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes a local-only logprobs response-shape observation for
  this exact bounded request; it is not generated-text correctness, tokenizer
  semantic correctness, prompt correctness, token identity or logprob value
  correctness, latency, throughput, long-context, production-readiness, or
  simpler-nv/vLLM integration evidence.

Detailed logprobs-contract evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_logprobs_contract_probe.md`.

Fresh remote H200 echo-contract evidence from this slice:

- The repo-owned response-contract probe script now has an `--echo-contract`
  mode that reuses the existing server lifecycle, health/model-list readiness
  checks, response-contract validation, cleanup, and local-only server
  boundary.
- The remote vLLM 0.23.0 completion request model was inspected before
  choosing this gate. It exposes explicit `echo`, `stop`, and
  `stop_token_ids` fields, and the non-streaming completion handler has
  explicit `request.echo` response handling.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, and a 180-second request timeout.
- The server started on `http://127.0.0.1:28130`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted one completion payload with `prompt=Hello`, `max_tokens=1`,
  `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, `stream=false`, and
  `echo=true`.
- The completion request returned HTTP 200 with exactly one response choice
  and structurally valid response-contract fields. The response reported
  `usage.prompt_tokens=1`, `usage.completion_tokens=1`, and
  `usage.total_tokens=2`.
- The explicit echo response-shape checks passed: the request carried
  `echo=true`, the request prompt was string-valued, and the response text
  started with the request prompt. The probe recorded only text length,
  generated suffix length, and a digest as opaque observations; it did not
  record raw generated text or compare the generated suffix to an expected
  answer.
- The probe exited 0 after 108.998 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes a local-only echo response-shape observation for this
  exact bounded request; it is not generated-text correctness, tokenizer
  semantic correctness, prompt correctness, latency, throughput,
  long-context, production-readiness, or simpler-nv/vLLM integration evidence.

Detailed echo-contract evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_echo_contract_probe.md`.

Fresh remote H200 stop-contract evidence from this slice:

- The repo-owned response-contract probe script now has a `--stop-contract`
  mode that reuses the existing server lifecycle, health/model-list readiness
  checks, response-contract validation, cleanup, and local-only server
  boundary.
- The remote vLLM 0.23.0 completion request model was inspected before
  choosing this gate. It exposes explicit `stop`, `stop_token_ids`, and
  `include_stop_str_in_output` fields, carries them into `SamplingParams`,
  and exposes `finish_reason` and `stop_reason` in completion responses.
- Source inspection did not provide a request-controlled way to force
  `deepseek-ai/DeepSeek-V4-Flash` to emit a chosen stop string or token
  without judging generated text or token identity, so this slice records
  stop-field acceptance and response-contract evidence rather than live
  stop-trigger evidence.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=4096`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 65-minute outer timeout, a
  2700-second readiness timeout, and a 180-second request timeout.
- The server started on `http://127.0.0.1:28131`, returned HTTP 200 from
  `/health`, returned `deepseek-ai/DeepSeek-V4-Flash` from `/v1/models`, and
  accepted one completion payload with `prompt=Hello`, `max_tokens=4`,
  `temperature=0.0`, `top_p=1.0`, `seed=0`, `n=1`, `stream=false`, one
  request-controlled stop string, one integer `stop_token_ids` entry, and
  `include_stop_str_in_output=false`.
- The completion request returned HTTP 200 with exactly one response choice
  and structurally valid response-contract fields. The response reported
  `usage.prompt_tokens=1`, `usage.completion_tokens=4`, and
  `usage.total_tokens=5`.
- The explicit stop-contract checks passed for request field shape and base
  response-contract validity. The result recorded `stop_trigger=not_asserted`;
  it did not assert stop marker presence, stop marker absence, token identity,
  or stop-token semantic correctness.
- The probe exited 0 after 111.779 seconds with `generation_attempted=true`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes local-only stop-field acceptance and response-contract
  evidence for this exact bounded request; it is not stop-trigger,
  generated-text correctness, tokenizer semantic correctness, prompt
  correctness, token identity or stop-token semantic correctness, latency,
  throughput, long-context, production-readiness, broad determinism, or
  simpler-nv/vLLM integration evidence.

Detailed stop-contract evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_stop_contract_probe.md`.

Fresh remote H200 64K context health evidence from this slice:

- A repo-owned server-health probe script was reused without sending a prompt:
  it starts `vllm serve`, binds only to `127.0.0.1`, checks `/health` and
  `/v1/models`, terminates the server process group, and reports remaining
  process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the complete
  ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB free
  on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=65536`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 50-minute outer timeout, and a
  2700-second readiness timeout.
- The server started on `http://127.0.0.1:28132`, returned HTTP 200 from
  `/health`, and returned `deepseek-ai/DeepSeek-V4-Flash` with
  `max_model_len=65536` from `/v1/models`.
- The vLLM server log reported `Using max model len 65536`, GPU KV cache
  size 673,323 tokens, and maximum concurrency 10.27x for 65,536 tokens per
  request.
- The probe exited 0 after 114.36 seconds with `generation_attempted=false`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at their pre-run
  memory baseline.
- The run establishes a local-only 64K server-health/model-list capacity
  gate only; it is not long-prompt, generated-text correctness, tokenizer
  semantic correctness, prompt correctness, latency, throughput,
  production-readiness, broad determinism, or simpler-nv/vLLM integration
  evidence.

Detailed 64K context health evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_64k_context_health_probe.md`.

Fresh remote H200 128K context health evidence from this slice:

- The repo-owned server-health probe script was reused without sending a
  prompt: it starts `vllm serve`, binds only to `127.0.0.1`, checks
  `/health` and `/v1/models`, terminates the server process group, and
  reports remaining process-group PIDs.
- The remote source tree was refreshed with `--sync`, preserving the
  complete ignored artifact directory and `.venv-vllm-probe`.
- The selected physical GPUs were again 1 and 7, exposed as exactly two
  visible devices with `CUDA_VISIBLE_DEVICES=1,7` and
  `tensor_parallel_size=2`. The fresh pre-run snapshot showed 141773 MiB
  free on GPU 1 and 116662 MiB free on GPU 7.
- The passing boundary used `max_model_len=131072`, `dtype=bfloat16`,
  `quantization=deepseek_v4_fp8`, `kv_cache_dtype=fp8`,
  `gpu_memory_utilization=0.78`, `enforce_eager=true`,
  `distributed_executor_backend=mp`, a 50-minute outer timeout, and a
  2700-second readiness timeout.
- The server started on `http://127.0.0.1:28133`, returned HTTP 200 from
  `/health`, and returned `deepseek-ai/DeepSeek-V4-Flash` with
  `max_model_len=131072` from `/v1/models`.
- The vLLM server log reported `Using max model len 131072`, GPU KV cache
  size 1,247,687 tokens, and maximum concurrency 9.52x for 131,072 tokens
  per request.
- The probe exited 0 after 101.546 seconds with `generation_attempted=false`.
- The probe sent `SIGTERM`, vLLM logged API server and engine shutdown, and
  the probe reported no remaining process-group PIDs. The immediate post-run
  selected-GPU memory snapshot showed GPU 1 and GPU 7 back at or near their
  pre-run memory baseline.
- The run establishes a local-only 128K server-health/model-list capacity
  gate only; it is not long-prompt, generated-text correctness, tokenizer
  semantic correctness, prompt correctness, latency, throughput,
  production-readiness, broad determinism, or simpler-nv/vLLM integration
  evidence.

Detailed 128K context health evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_128k_context_health_probe.md`.

## Serving Readiness State

The current remote state is complete through bounded two-H200 vLLM
server-startup, health/model-list readiness, one-token inference smoke, one
bounded response-contract probe, warmup/request-shape gates,
deterministic-repeat serving-semantics, one explicit logprobs response shape
probe, one explicit echo response-shape probe, one explicit stop-field
acceptance probe, and 64K plus 128K server-health/model-list capacity gates:

```text
remote_h200_reachable: yes
remote_vllm_environment: ready for weight-free import/config probes
remote_repo_relative_artifacts: complete for indexed shard presence
weight_free_require_vllm_probes: passed
artifact_require_probe: passed
weight_manifest_gate: complete
two_h200_vllm_model_load: passed under recorded 4096-token boundary
local_only_vllm_server_health: passed under recorded 4096-token boundary
local_only_vllm_64k_server_health: passed under recorded 65536-token boundary
local_only_vllm_128k_server_health: passed under recorded 131072-token boundary
one_token_inference_smoke: passed under recorded 4096-token boundary
response_contract_probe: passed under recorded 4096-token boundary
warmup_shape_probe: passed under recorded same-shape two-request boundary
request_shape_variation_probe: passed under recorded three-request boundary
serving_semantics_probe: passed under recorded deterministic-repeat boundary
logprobs_contract_probe: passed under recorded explicit logprob boundary
echo_contract_probe: passed under recorded explicit echo boundary
stop_contract_probe: passed under recorded explicit stop-field boundary
serving_readiness: bounded local-only response contract and warmup-shape
  variation plus deterministic-repeat serving-semantics, logprobs response
  shape, echo response-shape, stop-field acceptance observations, and 64K
  plus 128K server-health/model-list capacity gates; no correctness claims
```

This means the remote H200 environment has passed a local-only vLLM server
startup, one bounded local-only inference request, and one bounded
OpenAI-compatible completion response-contract check under the recorded
two-H200 boundary. It has also passed one bounded same-shape warmup/follow-up
observation and one bounded request-shape variation observation of the
selected Triton JIT warning string. It has now passed one bounded
deterministic-repeat serving-semantics observation that compares response
digests and accounting without recording or judging generated text. It has
also passed one bounded explicit `logprobs` / `prompt_logprobs` response-shape
observation without inspecting token identity or logprob values. It has also
passed one bounded explicit `echo=true` response-shape observation without
recording raw generated text or judging the generated suffix. It has also
passed one bounded explicit `stop` / `stop_token_ids` field-acceptance
observation without asserting stop triggering, marker presence, token
identity, or stop-token semantic correctness. It is still too early to claim
generated text correctness, token/logprob/stop-token semantic correctness,
long-prompt behavior, 256K context behavior, latency, throughput, production
readiness, broad determinism, or simpler-nv/vLLM kernel integration.

It has also passed local-only 64K and 128K server-health/model-list capacity
gates without sending a long prompt or attempting generation.

## Next Gate

The next PR-sized capacity gate can attempt 256K server-health/model-list
readiness under the same local-only, no-long-prompt contract. That later gate
needs its own command, resource plan, expected failure mode, and non-claims,
and it should stay separate from generated-text correctness, throughput,
latency, and production-readiness claims.

## Non-Claims

- This is not prompt, tokenizer semantic, correct-text, long-prompt, 256K
  context, throughput, latency, or production readiness evidence.
- This is not generated-text correctness evidence.
- This is not token identity or logprob value correctness evidence.
- This is not stop-trigger or stop-token semantic correctness evidence.
- This is not broad serving determinism evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.
