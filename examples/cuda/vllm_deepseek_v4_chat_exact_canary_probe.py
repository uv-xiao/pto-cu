#!/usr/bin/env python3
"""Bounded vLLM chat-completions exact-output canary for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
HEALTH_PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_server_health_probe.py"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_EXACT_CANARY_28149"
DEFAULT_MAX_MODEL_LEN = 262_144
NORMALIZATION_RULE = (
    "strip leading/trailing whitespace, then strip one surrounding Markdown "
    "code fence when the whole output is fenced"
)
NON_CLAIMS = [
    "not general generated-text correctness evidence",
    "not semantic correctness evidence",
    "not throughput or latency evidence",
    "not production-readiness evidence",
    "not broad determinism evidence",
    "not simpler-nv or vLLM integration evidence",
]
CONTRACT_CHECKS = [
    "HTTP 200 from /health",
    "HTTP 200 from /v1/models",
    "model list includes served model and max_model_len=262144",
    "HTTP 200 from one non-streaming /v1/chat/completions request",
    "top-level chat completion response is a JSON object",
    "response model field matches served model when returned",
    "exactly one response choice object",
    "first choice exposes assistant message content and finish_reason fields",
    "normalized assistant content equals the expected canary string",
    "usage prompt/completion/total token fields are internally consistent when returned",
    "usage.completion_tokens within request max_tokens",
    "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
    "raw prompt text is not recorded",
    "raw request payload is not recorded",
    "token ID arrays are not recorded",
    "server process group cleanup leaves no remaining PIDs",
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


def _runtime_versions() -> dict[str, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        return health_probe._runtime_versions()


def normalize_generated_output_for_exact_match(text: str) -> str:
    """Apply the narrow exact-match normalization used by the probe."""
    normalized = text.strip()
    lines = normalized.splitlines()
    if (
        len(lines) >= 2
        and lines[0].startswith("```")
        and lines[-1].strip() == "```"
    ):
        normalized = "\n".join(lines[1:-1]).strip()
    return normalized


def _validate_request_bounds(*, max_tokens: int, expected_answer: str) -> None:
    if max_tokens < 1 or max_tokens > DEFAULT_MAX_TOKENS:
        raise ValueError(
            "chat canary requests require "
            f"1 <= max_tokens <= {DEFAULT_MAX_TOKENS}"
        )
    if not expected_answer:
        raise ValueError("expected chat canary answer must be non-empty")


def _limits(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    expected_answer: str,
) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "stream": False,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "expected_answer": expected_answer,
        "match_mode": "exact",
        "normalization": NORMALIZATION_RULE,
        "message_count": 2,
        "message_roles": ["system", "user"],
        "prompt_text_recording": False,
        "payload_recording": False,
        "generated_text_recording": False,
    }


def build_chat_canary_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
) -> dict[str, Any]:
    _validate_request_bounds(max_tokens=max_tokens, expected_answer=expected_answer)
    payload = {
        "model": served_model_name,
        "messages": [
            {
                "role": "system",
                "content": "Return only the exact requested canary string.",
            },
            {
                "role": "user",
                "content": (
                    "Return exactly this string and nothing else: "
                    f"{expected_answer}"
                ),
            },
        ],
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
        "limits": _limits(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            expected_answer=expected_answer,
        ),
    }


def build_planned_chat_canary_request(
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
) -> dict[str, Any]:
    _validate_request_bounds(max_tokens=max_tokens, expected_answer=expected_answer)
    return {
        "endpoint": DEFAULT_ENDPOINT,
        "limits": _limits(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            expected_answer=expected_answer,
        ),
    }


def review_safe_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": request["endpoint"],
        "limits": request["limits"],
        "prompt_text_recorded": False,
        "payload_recorded": False,
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


def _response_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    usage = payload.get("usage")
    safe_top_level_keys = {
        "choices",
        "created",
        "id",
        "kv_transfer_params",
        "model",
        "object",
        "service_tier",
        "system_fingerprint",
        "usage",
    }
    safe_choice_keys = {"finish_reason", "index", "message"}
    safe_message_keys = {"role", "content"}
    return {
        "top_level_review_safe_keys": sorted(
            key for key in payload.keys() if key in safe_top_level_keys
        ),
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "first_choice_review_safe_keys": sorted(
            key for key in first_choice.keys() if key in safe_choice_keys
        )
        if isinstance(first_choice, dict)
        else [],
        "message_review_safe_keys": sorted(
            key for key in message.keys() if key in safe_message_keys
        )
        if isinstance(message, dict)
        else [],
        "usage_keys": sorted(usage.keys()) if isinstance(usage, dict) else [],
    }


def validate_chat_canary_response(
    payload: Any,
    *,
    request: dict[str, Any],
    served_model_name: str,
    expected_answer: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "expected_answer": expected_answer,
        "match_mode": "exact",
        "normalization": NORMALIZATION_RULE,
        "response_shape": _response_shape(payload),
        "checks": {},
    }
    if not isinstance(payload, dict):
        result["checks"]["payload_object"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_payload_shape",
            message="chat completion response payload is not a JSON object",
        )
    result["checks"]["payload_object"] = "passed"

    model = payload.get("model")
    if model is None:
        result["checks"]["model_field"] = "not_returned"
    elif model != served_model_name:
        result["checks"]["model_field"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_model_mismatch",
            message=(
                f"response model {model!r} does not match served model "
                f"{served_model_name!r}"
            ),
        )
    else:
        result["checks"]["model_field"] = "passed"
        result["model"] = model

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        result["checks"]["choice_count"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_choice_shape",
            message="chat completion response must contain exactly one object choice",
        )
    result["checks"]["choice_count"] = "passed"
    result["choice_count"] = 1

    choice = choices[0]
    message = choice.get("message")
    if (
        "finish_reason" not in choice
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        result["checks"]["choice_fields"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_choice_fields",
            message=(
                "chat completion choice must include finish_reason and an "
                "assistant message with string content"
            ),
        )
    result["checks"]["choice_fields"] = "passed"
    result["finish_reason"] = choice.get("finish_reason")

    content = message["content"]
    normalized_output = normalize_generated_output_for_exact_match(content)
    result["normalized_output_length_chars"] = len(normalized_output)
    result["normalized_output_equals_expected"] = normalized_output == expected_answer
    if normalized_output != expected_answer:
        result["checks"]["expected_answer_exact"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_expected_answer_not_exact",
            message="normalized assistant content did not exactly equal the expected canary",
        )
    result["checks"]["expected_answer_exact"] = "passed"

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_usage_shape",
            message="chat completion response must include a usage object",
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if not all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_usage_shape",
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
            category="chat_canary_completion_bound",
            message=(
                f"usage.completion_tokens={completion_tokens} exceeds request "
                f"max_tokens={max_tokens}"
            ),
        )
    result["checks"]["usage_completion_bound"] = "passed"

    if total_tokens < prompt_tokens + completion_tokens:
        result["checks"]["usage_total_tokens"] = "failed"
        return _fail_contract(
            result,
            category="chat_canary_usage_total_tokens",
            message=(
                "usage.total_tokens must be greater than or equal to "
                "usage.prompt_tokens + usage.completion_tokens"
            ),
        )
    result["checks"]["usage_total_tokens"] = "passed"
    return result


def send_chat_canary_request(
    *,
    port: int,
    request: dict[str, Any],
    served_model_name: str,
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
) -> dict[str, Any]:
    endpoint = request["endpoint"]
    scheme = "http"
    url = f"{scheme}://{health_probe.LOCAL_HOST}:{port}{endpoint}"
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
                    "chat_completion_http_status",
                    f"chat completion endpoint returned HTTP {status}",
                )
                return result
            payload = _read_json_response(response)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["failure"] = _failure_payload(
            "chat_completion_http_error",
            _exception_summary(exc),
        )
        return result
    except TimeoutError as exc:
        result["failure"] = _failure_payload(
            "chat_completion_timeout",
            _exception_summary(exc),
        )
        return result
    except Exception as exc:
        result["failure"] = _failure_payload(
            "chat_completion_request_error",
            _exception_summary(exc),
        )
        return result

    validation = validate_chat_canary_response(
        payload,
        request=request,
        served_model_name=served_model_name,
        expected_answer=expected_answer,
    )
    result["validation"] = validation
    result["response_shape"] = validation["response_shape"]
    result["status"] = validation["status"]
    if validation["status"] != "passed":
        result["failure"] = validation["failure"]
    else:
        result["observation"] = {
            key: validation[key]
            for key in (
                "expected_answer",
                "match_mode",
                "normalization",
                "finish_reason",
                "normalized_output_length_chars",
                "normalized_output_equals_expected",
                "usage",
            )
            if key in validation
        }
    return result


def run_probe(
    *,
    artifact_dir: Path = health_probe.DEFAULT_ARTIFACT_DIR,
    vllm_bin: Path = health_probe.DEFAULT_VLLM_BIN,
    port: int | None = None,
    server_log: Path | None = None,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_port = port if port is not None else health_probe.choose_local_port()
    if selected_port <= 0 or selected_port > 65535:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_port", f"invalid port: {selected_port}"),
            "generation_attempted": False,
            "prompt_sent": False,
            "non_claims": NON_CLAIMS,
        }
    try:
        request = (
            build_planned_chat_canary_request(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                expected_answer=expected_answer,
            )
            if dry_run
            else build_chat_canary_request(
                served_model_name=served_model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                expected_answer=expected_answer,
            )
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_request", str(exc)),
            "generation_attempted": False,
            "prompt_sent": False,
            "non_claims": NON_CLAIMS,
        }

    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-chat-exact-canary-probe"
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
        "server_command_summary": {
            "argv0": health_probe._command_path(vllm_bin),
            "subcommand": "serve",
            "host": health_probe.LOCAL_HOST,
            "port": selected_port,
            "served_model_name": served_model_name,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "distributed_executor_backend": distributed_executor_backend,
            "enforce_eager": enforce_eager,
            "trust_remote_code": trust_remote_code,
        },
        "server_host": health_probe.LOCAL_HOST,
        "server_port": selected_port,
        "server_log": health_probe._display_path(log_path),
        "endpoints": ["/health", "/v1/models", DEFAULT_ENDPOINT],
        "request": review_safe_request(request),
        "contract_checks": CONTRACT_CHECKS,
        "request_timeout_seconds": request_timeout_seconds,
        "runtime_versions": _runtime_versions(),
        "gpu_memory_before": health_probe._query_nvidia_smi_memory(),
        "generation_attempted": False,
        "prompt_sent": False,
        "non_claims": NON_CLAIMS,
    }
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
        completion = send_chat_canary_request(
            port=selected_port,
            request=request,
            served_model_name=served_model_name,
            expected_answer=expected_answer,
            timeout_seconds=request_timeout_seconds,
        )
        result["generation_attempted"] = True
        result["prompt_sent"] = True
        result["chat_completion"] = completion
        result["status"] = completion["status"]
        if completion["status"] != "passed":
            result["failure"] = completion.get(
                "failure",
                _failure_payload("chat_canary_failed", "chat canary failed"),
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
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
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
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-answer", default=DEFAULT_EXPECTED_ANSWER)
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
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        expected_answer=args.expected_answer,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
