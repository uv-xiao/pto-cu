#!/usr/bin/env python3
"""Bounded vLLM completion response-contract probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
HEALTH_PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_server_health_probe.py"
DEFAULT_PROMPT = "Hello"
DEFAULT_ENDPOINT = "/v1/completions"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 4
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
MAX_ALLOWED_TOKENS = 16
DEFAULT_LOGPROBS = 1
DEFAULT_PROMPT_LOGPROBS = 1
MAX_ALLOWED_LOGPROBS = 5
NON_CLAIMS = [
    "not generated-text correctness evidence",
    "not tokenizer semantic correctness evidence",
    "not prompt correctness evidence",
    "not 256K context evidence",
    "not throughput or latency evidence",
    "not production-readiness evidence",
    "not simpler-nv or vLLM kernel integration evidence",
]


HttpPost = Callable[[str, dict[str, Any], float], Any]


def _load_health_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_server_health_probe", HEALTH_PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {HEALTH_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


health_probe = _load_health_probe()


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _read_json_response(response: Any) -> Any:
    data = response.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def _http_post(url: str, payload: dict[str, Any], timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def build_contract_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if max_tokens < 1 or max_tokens > MAX_ALLOWED_TOKENS:
        raise ValueError(f"response-contract requests require 1..{MAX_ALLOWED_TOKENS} tokens")
    payload: dict[str, Any] = {
        "model": served_model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "n": 1,
        "stream": False,
    }
    return {
        "endpoint": DEFAULT_ENDPOINT,
        "payload": payload,
        "limits": {
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt),
            "stream": False,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }


def build_logprobs_contract_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    logprobs: int = DEFAULT_LOGPROBS,
    prompt_logprobs: int = DEFAULT_PROMPT_LOGPROBS,
) -> dict[str, Any]:
    if logprobs < 1 or logprobs > MAX_ALLOWED_LOGPROBS:
        raise ValueError(f"logprobs-contract requests require 1..{MAX_ALLOWED_LOGPROBS}")
    if prompt_logprobs < 1 or prompt_logprobs > MAX_ALLOWED_LOGPROBS:
        raise ValueError(
            f"logprobs-contract requests require prompt_logprobs "
            f"1..{MAX_ALLOWED_LOGPROBS}"
        )
    request = build_contract_request(
        served_model_name=served_model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    request["payload"]["logprobs"] = logprobs
    request["payload"]["prompt_logprobs"] = prompt_logprobs
    request["limits"]["logprobs"] = logprobs
    request["limits"]["prompt_logprobs"] = prompt_logprobs
    return request


def build_echo_contract_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    request = build_contract_request(
        served_model_name=served_model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    request["payload"]["echo"] = True
    request["limits"]["echo"] = True
    return request


def _response_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    usage = payload.get("usage")
    return {
        "top_level_keys": sorted(payload.keys()),
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "first_choice_keys": sorted(first_choice.keys())
        if isinstance(first_choice, dict)
        else [],
        "usage_keys": sorted(usage.keys()) if isinstance(usage, dict) else [],
    }


def _completion_observation(payload: dict[str, Any]) -> dict[str, Any]:
    choice = payload["choices"][0]
    text = choice["text"]
    usage = payload["usage"]
    return {
        "model": payload["model"],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "text_length_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }


def _fail_contract(
    result: dict[str, Any],
    *,
    category: str,
    message: str,
) -> dict[str, Any]:
    result["status"] = "failed"
    result["failure"] = _failure_payload(category, message)
    return result


def validate_completion_contract(
    payload: Any,
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "response_shape": _response_shape(payload),
        "checks": {},
    }
    if not isinstance(payload, dict):
        return _fail_contract(
            result,
            category="response_contract_payload_shape",
            message="completion response payload is not a JSON object",
        )

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        result["checks"]["choice_count"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_choice_shape",
            message="completion response must contain exactly one object choice",
        )
    result["checks"]["choice_count"] = "passed"
    result["choice_count"] = 1

    choice = choices[0]
    if "finish_reason" not in choice or not isinstance(choice.get("text"), str):
        result["checks"]["choice_fields"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_choice_fields",
            message="completion choice must include finish_reason and text",
        )
    result["checks"]["choice_fields"] = "passed"

    if not isinstance(payload.get("model"), str) or not payload["model"]:
        result["checks"]["model_field"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_model_field",
            message="completion response must include a non-empty model field",
        )
    result["checks"]["model_field"] = "passed"
    result["model"] = payload["model"]

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_usage_shape",
            message="completion response must include a usage object",
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if not all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_usage_shape",
            message="usage.prompt_tokens, completion_tokens, and total_tokens must be integers",
        )
    result["checks"]["usage_shape"] = "passed"
    result["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    max_tokens = request["limits"]["max_tokens"]
    if completion_tokens < 0 or completion_tokens > max_tokens:
        result["checks"]["usage_completion_bound"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_completion_bound",
            message=(
                f"usage.completion_tokens={completion_tokens} exceeds request "
                f"max_tokens={max_tokens}"
            ),
        )
    result["checks"]["usage_completion_bound"] = "passed"

    if total_tokens < prompt_tokens:
        result["checks"]["usage_total_tokens"] = "failed"
        return _fail_contract(
            result,
            category="response_contract_usage_total_tokens",
            message="usage.total_tokens must be greater than or equal to usage.prompt_tokens",
        )
    result["checks"]["usage_total_tokens"] = "passed"

    token_ids = choice.get("token_ids")
    if token_ids is None:
        result["checks"]["token_ids"] = "not_present"
    elif not isinstance(token_ids, list) or len(token_ids) != completion_tokens:
        result["checks"]["token_ids"] = "failed"
        token_count = len(token_ids) if isinstance(token_ids, list) else "non-list"
        return _fail_contract(
            result,
            category="response_contract_token_ids_mismatch",
            message=(
                f"choice.token_ids length {token_count} is not compatible with "
                f"usage.completion_tokens={completion_tokens}"
            ),
        )
    else:
        result["checks"]["token_ids"] = "passed"
        result["token_ids_count"] = len(token_ids)

    return result


def _is_logprob_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _validate_completion_logprobs_shape(
    logprobs: Any,
    *,
    completion_tokens: int,
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(logprobs, dict):
        return "completion logprobs must be a JSON object", None
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    text_offset = logprobs.get("text_offset")
    fields = (tokens, token_logprobs, top_logprobs, text_offset)
    if not all(isinstance(field, list) for field in fields):
        return (
            "completion logprobs tokens, token_logprobs, top_logprobs, "
            "and text_offset must be lists"
        ), None
    field_lengths = {
        "tokens": len(tokens),
        "token_logprobs": len(token_logprobs),
        "top_logprobs": len(top_logprobs),
        "text_offset": len(text_offset),
    }
    if set(field_lengths.values()) != {completion_tokens}:
        return (
            f"completion logprobs list lengths {field_lengths} must match "
            f"usage.completion_tokens={completion_tokens}"
        ), None
    if not all(isinstance(token, str) for token in tokens):
        return "completion logprobs tokens must be strings", None
    if not all(value is None or _is_logprob_number(value) for value in token_logprobs):
        return "completion token_logprobs values must be numbers or null", None
    if not all(isinstance(value, int) for value in text_offset):
        return "completion text_offset values must be integers", None
    if not all(value is None or isinstance(value, dict) for value in top_logprobs):
        return "completion top_logprobs values must be objects or null", None
    return "", {
        "completion_token_count": len(tokens),
        "top_logprobs_entry_count": len(top_logprobs),
    }


def _validate_prompt_logprobs_shape(
    prompt_logprobs: Any,
    *,
    prompt_tokens: int,
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(prompt_logprobs, list):
        return "prompt_logprobs must be a list", None
    if len(prompt_logprobs) > prompt_tokens:
        return (
            f"prompt_logprobs length {len(prompt_logprobs)} exceeds "
            f"usage.prompt_tokens={prompt_tokens}"
        ), None
    if not all(value is None or isinstance(value, dict) for value in prompt_logprobs):
        return "prompt_logprobs entries must be objects or null", None
    non_null_entries = sum(1 for value in prompt_logprobs if value is not None)
    return "", {
        "prompt_logprobs_count": len(prompt_logprobs),
        "prompt_logprobs_non_null_count": non_null_entries,
    }


def validate_logprobs_contract(
    payload: Any,
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    result = validate_completion_contract(payload, request=request)
    result["logprobs"] = {}
    if result["status"] != "passed":
        result["checks"]["base_completion_contract"] = "failed"
        return result
    result["checks"]["base_completion_contract"] = "passed"
    assert isinstance(payload, dict)
    choice = payload["choices"][0]

    completion_error, completion_summary = _validate_completion_logprobs_shape(
        choice.get("logprobs"),
        completion_tokens=result["usage"]["completion_tokens"],
    )
    if completion_error:
        result["checks"]["completion_logprobs_shape"] = "failed"
        return _fail_contract(
            result,
            category="logprobs_contract_completion_shape",
            message=completion_error,
        )
    result["checks"]["completion_logprobs_shape"] = "passed"
    result["logprobs"].update(completion_summary or {})

    prompt_error, prompt_summary = _validate_prompt_logprobs_shape(
        choice.get("prompt_logprobs"),
        prompt_tokens=result["usage"]["prompt_tokens"],
    )
    if prompt_error:
        result["checks"]["prompt_logprobs_shape"] = "failed"
        category = (
            "logprobs_contract_prompt_bound"
            if "exceeds usage.prompt_tokens" in prompt_error
            else "logprobs_contract_prompt_shape"
        )
        return _fail_contract(result, category=category, message=prompt_error)
    result["checks"]["prompt_logprobs_shape"] = "passed"
    result["logprobs"].update(prompt_summary or {})
    return result


def validate_echo_contract(
    payload: Any,
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    result = validate_completion_contract(payload, request=request)
    result["echo"] = {}
    if result["status"] != "passed":
        result["checks"]["base_completion_contract"] = "failed"
        return result
    result["checks"]["base_completion_contract"] = "passed"

    request_payload = request.get("payload")
    prompt = request_payload.get("prompt") if isinstance(request_payload, dict) else None
    if not isinstance(prompt, str):
        result["checks"]["echo_request_prompt"] = "failed"
        return _fail_contract(
            result,
            category="echo_contract_request_prompt",
            message="echo-contract request must use a string prompt",
        )
    if request_payload.get("echo") is not True:
        result["checks"]["echo_request_field"] = "failed"
        return _fail_contract(
            result,
            category="echo_contract_request_field",
            message="echo-contract request must set echo=true",
        )

    assert isinstance(payload, dict)
    choice_text = payload["choices"][0]["text"]
    if not choice_text.startswith(prompt):
        result["checks"]["echo_prompt_prefix"] = "failed"
        return _fail_contract(
            result,
            category="echo_contract_prompt_prefix",
            message="echo completion text must start with the request prompt",
        )
    result["checks"]["echo_request_prompt"] = "passed"
    result["checks"]["echo_request_field"] = "passed"
    result["checks"]["echo_prompt_prefix"] = "passed"
    result["echo"] = {
        "prompt_chars": len(prompt),
        "text_length_chars": len(choice_text),
        "generated_suffix_chars": max(0, len(choice_text) - len(prompt)),
    }
    return result


def send_contract_request(
    *,
    port: int,
    request: dict[str, Any],
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
    validator: Callable[..., dict[str, Any]] = validate_completion_contract,
) -> dict[str, Any]:
    endpoint = request["endpoint"]
    url = f"http://{health_probe.LOCAL_HOST}:{port}{endpoint}"
    result: dict[str, Any] = {
        "status": "failed",
        "endpoint": endpoint,
        "url": url,
        "request_limits": request["limits"],
    }
    try:
        with http_post(url, request["payload"], timeout_seconds) as response:
            status = int(getattr(response, "status", 0))
            result["http_status"] = status
            if status != 200:
                result["failure"] = _failure_payload(
                    "completion_http_status",
                    f"completion endpoint returned HTTP {status}",
                )
                return result
            payload = _read_json_response(response)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["failure"] = _failure_payload(
            "completion_http_error",
            _exception_summary(exc),
        )
        return result
    except TimeoutError as exc:
        result["failure"] = _failure_payload(
            "completion_timeout",
            _exception_summary(exc),
        )
        return result
    except Exception as exc:
        result["failure"] = _failure_payload(
            "completion_request_error",
            _exception_summary(exc),
        )
        return result

    contract = validator(payload, request=request)
    result["contract"] = contract
    result["response_shape"] = contract["response_shape"]
    result["status"] = contract["status"]
    if contract["status"] != "passed":
        result["failure"] = contract["failure"]
    else:
        result["observation"] = _completion_observation(payload)
    return result


def send_logprobs_contract_request(
    *,
    port: int,
    request: dict[str, Any],
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
) -> dict[str, Any]:
    return send_contract_request(
        port=port,
        request=request,
        timeout_seconds=timeout_seconds,
        http_post=http_post,
        validator=validate_logprobs_contract,
    )


def send_echo_contract_request(
    *,
    port: int,
    request: dict[str, Any],
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
) -> dict[str, Any]:
    return send_contract_request(
        port=port,
        request=request,
        timeout_seconds=timeout_seconds,
        http_post=http_post,
        validator=validate_echo_contract,
    )


def run_probe(
    *,
    artifact_dir: Path = health_probe.DEFAULT_ARTIFACT_DIR,
    vllm_bin: Path = health_probe.DEFAULT_VLLM_BIN,
    port: int | None = None,
    server_log: Path | None = None,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    max_model_len: int = health_probe.DEFAULT_MAX_MODEL_LEN,
    tensor_parallel_size: int = health_probe.DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype: str = health_probe.DEFAULT_DTYPE,
    quantization: str | None = health_probe.DEFAULT_QUANTIZATION,
    kv_cache_dtype: str = health_probe.DEFAULT_KV_CACHE_DTYPE,
    gpu_memory_utilization: float = health_probe.DEFAULT_GPU_MEMORY_UTILIZATION,
    distributed_executor_backend: str = health_probe.DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    enforce_eager: bool = True,
    trust_remote_code: bool = False,
    timeout_seconds: float = health_probe.DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = health_probe.DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    terminate_timeout_seconds: float = health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    echo_contract: bool = False,
    logprobs_contract: bool = False,
    logprobs: int = DEFAULT_LOGPROBS,
    prompt_logprobs: int = DEFAULT_PROMPT_LOGPROBS,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_port = port if port is not None else health_probe.choose_local_port()
    if selected_port <= 0 or selected_port > 65535:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_port", f"invalid port: {selected_port}"),
            "generation_attempted": False,
            "non_claims": NON_CLAIMS,
        }
    if echo_contract and logprobs_contract:
        return {
            "status": "failed",
            "failure": _failure_payload(
                "invalid_request",
                "choose only one explicit contract mode: echo or logprobs",
            ),
            "generation_attempted": False,
            "non_claims": NON_CLAIMS,
        }
    try:
        if echo_contract:
            request_builder = build_echo_contract_request
        elif logprobs_contract:
            request_builder = build_logprobs_contract_request
        else:
            request_builder = build_contract_request
        request_kwargs = {
            "served_model_name": served_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        if logprobs_contract:
            request_kwargs["logprobs"] = logprobs
            request_kwargs["prompt_logprobs"] = prompt_logprobs
        contract_request = request_builder(**request_kwargs)
    except ValueError as exc:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_request", str(exc)),
            "generation_attempted": False,
            "non_claims": NON_CLAIMS,
        }
    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-response-contract-probe"
        / f"server-{selected_port}.log"
    )
    command = health_probe.build_server_command(
        vllm_bin=vllm_bin,
        artifact_dir=artifact_dir,
        port=selected_port,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
    )
    result: dict[str, Any] = {
        "status": "planned" if dry_run else "failed",
        "server_command": command,
        "server_host": health_probe.LOCAL_HOST,
        "server_port": selected_port,
        "server_log": health_probe._display_path(log_path),
        "endpoints": ["/health", "/v1/models", DEFAULT_ENDPOINT],
        "contract_mode": (
            "echo" if echo_contract else "logprobs" if logprobs_contract else "response"
        ),
        "contract_request": contract_request,
        "contract_checks": [
            "HTTP 200 from /health",
            "HTTP 200 from /v1/models",
            "HTTP 200 from /v1/completions",
            "exactly one response choice",
            "choice text and finish_reason fields present",
            "response model field present",
            "usage prompt/completion/total token fields present",
            "usage.completion_tokens within request max_tokens",
            "usage.total_tokens >= usage.prompt_tokens",
            "token_ids length matches usage.completion_tokens when present",
            "local-only non-streaming bounded request with explicit sampler settings",
            "server process group cleanup leaves no remaining PIDs",
        ],
        "request_timeout_seconds": request_timeout_seconds,
        "runtime_versions": health_probe._runtime_versions(),
        "gpu_memory_before": health_probe._query_nvidia_smi_memory(),
        "generation_attempted": False,
        "non_claims": NON_CLAIMS,
    }
    if logprobs_contract:
        result["contract_checks"].extend(
            [
                "explicit logprobs and prompt_logprobs request fields",
                "choice.logprobs exposes list-valued completion logprob fields",
                "completion logprob list lengths match usage.completion_tokens",
                "choice.prompt_logprobs is list-valued and bounded by usage.prompt_tokens",
            ]
        )
    if echo_contract:
        result["contract_checks"].extend(
            [
                "explicit echo=true request field",
                "echo response text starts with request prompt",
                "raw generated text is not recorded or checked for correctness",
            ]
        )
    if dry_run:
        return result

    started = time.monotonic()
    process = None
    log_file = None
    try:
        process, log_file = health_probe._start_server(command, log_path)
        result["server_pid"] = process.pid
        readiness = health_probe.poll_readiness(
            port=selected_port,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            request_timeout_seconds=5.0,
        )
        result["readiness"] = readiness
        if readiness["status"] != "passed":
            result["status"] = "failed"
            result["failure"] = readiness.get(
                "failure",
                _failure_payload("readiness_failed", "server readiness failed"),
            )
            return result
        if echo_contract:
            send_request = send_echo_contract_request
        elif logprobs_contract:
            send_request = send_logprobs_contract_request
        else:
            send_request = send_contract_request
        completion = send_request(
            port=selected_port,
            request=contract_request,
            timeout_seconds=request_timeout_seconds,
        )
        result["generation_attempted"] = True
        result["completion"] = completion
        result["status"] = completion["status"]
        if completion["status"] != "passed":
            result["failure"] = completion.get(
                "failure",
                _failure_payload("completion_contract_failed", "completion contract failed"),
            )
        if process.poll() is not None and result["status"] != "passed":
            result["failure"] = _failure_payload(
                "server_exited_during_probe",
                f"server process exited with {process.returncode}",
            )
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = _failure_payload("probe_error", _exception_summary(exc))
    finally:
        if process is not None:
            result["cleanup"] = health_probe.cleanup_process(
                process,
                terminate_timeout_seconds=terminate_timeout_seconds,
            )
            if result["cleanup"]["status"] != "passed":
                result["status"] = "failed"
                result["failure"] = _failure_payload(
                    "cleanup_failed",
                    "server process group still had remaining processes after cleanup",
                )
        if log_file is not None:
            log_file.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["gpu_memory_after"] = health_probe._query_nvidia_smi_memory()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=health_probe.DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vllm-bin", type=Path, default=health_probe.DEFAULT_VLLM_BIN)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--served-model-name", default=health_probe.DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=health_probe.DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=health_probe.DEFAULT_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument("--dtype", default=health_probe.DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=health_probe.DEFAULT_QUANTIZATION)
    parser.add_argument("--kv-cache-dtype", default=health_probe.DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=health_probe.DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default=health_probe.DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=health_probe.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=health_probe.DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--terminate-timeout-seconds",
        type=float,
        default=health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--echo-contract", action="store_true")
    parser.add_argument("--logprobs-contract", action="store_true")
    parser.add_argument("--logprobs", type=int, default=DEFAULT_LOGPROBS)
    parser.add_argument("--prompt-logprobs", type=int, default=DEFAULT_PROMPT_LOGPROBS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(
        artifact_dir=args.artifact_dir,
        vllm_bin=args.vllm_bin,
        port=args.port,
        server_log=args.server_log,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        terminate_timeout_seconds=args.terminate_timeout_seconds,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        echo_contract=args.echo_contract,
        logprobs_contract=args.logprobs_contract,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
