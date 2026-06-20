# vLLM Remote Chat 256K Needle Exact Probe

## Summary

- status: passed
- PROBE_EXIT_STATUS=0
- source refresh: remote tree sync of this branch's final probe code
- remote vLLM environment: preserved `.venv-vllm-probe`
- CUDA_VISIBLE_DEVICES=1,7
- server_host: 127.0.0.1
- server_port: 28151
- endpoint: /v1/chat/completions
- model: deepseek-ai/DeepSeek-V4-Flash
- vllm: 0.23.0
- torch: 2.11.0+cu130
- torch CUDA: 13.0
- python: 3.12.3
- elapsed_seconds: 129.778

## Command Boundary

````text
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc '<bounded command>'

CUDA_VISIBLE_DEVICES=1,7
VLLM_NO_USAGE_STATS=1
.venv-vllm-probe/bin/python
examples/cuda/vllm_deepseek_v4_chat_256k_needle_exact_probe.py
--artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
--vllm-bin .venv-vllm-probe/bin/vllm
--port 28151
--server-log tmp/vllm-chat-256k-needle-exact-probe/server-28151.log
--max-model-len 262144
--tensor-parallel-size 2
--dtype bfloat16
--quantization deepseek_v4_fp8
--kv-cache-dtype fp8
--gpu-memory-utilization 0.78
--distributed-executor-backend mp
--enforce-eager
--timeout-seconds 2700
--poll-interval-seconds 10
--request-timeout-seconds 180
--terminate-timeout-seconds 60
--target-prompt-tokens 255800
--max-tokens 64
--temperature 0.0
--top-p 1.0
--seed 0
--expected-answer PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151
stop_sequence: \n```
````

## Request Limits

- max_model_len=262144
- tensor_parallel_size=2
- dtype=bfloat16
- quantization=deepseek_v4_fp8
- kv_cache_dtype=fp8
- gpu_memory_utilization=0.78
- distributed_executor_backend=mp
- enforce_eager=true
- target_prompt_tokens=255800
- tokenizer_accounting: transformers.AutoTokenizer fallback encode estimate
- actual_prompt_tokens: not available from local tokenizer accounting
- prompt_chars: 1233751
- prompt_unit_chars: 82
- max_tokens=64
- temperature=0.0
- top_p=1.0
- seed=0
- stream: false
- n: 1
- message_count: 2
- message_roles: system,user
- needle_position: middle
- needle_occurrences: 1
- filler_units_before_needle: 7520
- filler_units_after_needle: 7521
- expected_answer: PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151
- match_mode: exact
- normalization: strip leading/trailing whitespace, then strip one surrounding
  Markdown code fence when the whole output is fenced

## Endpoint Result

- HTTP status: 200
- finish_reason: stop
- normalized_output_equals_expected: true
- normalized_output_length_chars: 37
- exact_match: true
- expected_answer_exact: passed
- usage.prompt_tokens: 255795
- usage.completion_tokens: 18
- usage.total_tokens: 255813
- usage_prompt_tokens_match: not_available
- usage_completion_bound: passed
- usage_total_tokens: passed
- response_shape.choice_count: 1
- response_shape.first_choice_review_safe_keys: finish_reason,index,message
- response_shape.message_review_safe_keys: content,role
- response_shape.usage_keys:
  completion_tokens,prompt_tokens,prompt_tokens_details,total_tokens

## Readiness And Cleanup

- /health HTTP status: 200
- /v1/models HTTP status: 200
- served model listed: deepseek-ai/DeepSeek-V4-Flash
- served model max_model_len: 262144
- readiness_contract: passed
- cleanup.status: passed
- cleanup.terminated: true
- cleanup.killed: false
- cleanup.returncode_after_cleanup: 0
- remaining_process_group_pids: []

## Review Safety

- raw prompt text is not recorded
- raw request payload is not recorded
- raw generated text is not recorded
- token ID arrays are not recorded
- logprob values are not recorded
- generated-text digests are not recorded
- model artifact contents are not recorded
- private absolute paths are not recorded

## Non-Claims

- This is one local-only OpenAI-compatible chat-completions synthetic needle
  exact-output gate under the recorded two-H200 vLLM boundary.
- This is not general generated-text correctness evidence.
- This is not semantic correctness evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
