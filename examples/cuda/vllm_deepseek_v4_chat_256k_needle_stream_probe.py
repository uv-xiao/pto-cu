#!/usr/bin/env python3
"""Streaming vLLM chat 256K synthetic needle exact probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import urllib.error
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_exact_probe.py"
)
DEFAULT_PORT = 28_153
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_STREAM_OK_28153"
DEFAULT_STOP_SEQUENCES = ("\n```",)


def _load_base_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_exact_probe",
        BASE_PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {BASE_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


base_probe = _load_base_probe()
health_probe = base_probe.health_probe
long_prompt_contract = base_probe.long_prompt_contract

DEFAULT_ENDPOINT = base_probe.DEFAULT_ENDPOINT
DEFAULT_TARGET_PROMPT_TOKENS = base_probe.DEFAULT_TARGET_PROMPT_TOKENS
DEFAULT_REQUEST_TIMEOUT_SECONDS = base_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS
DEFAULT_MAX_TOKENS = base_probe.DEFAULT_MAX_TOKENS
DEFAULT_TEMPERATURE = base_probe.DEFAULT_TEMPERATURE
DEFAULT_TOP_P = base_probe.DEFAULT_TOP_P
DEFAULT_SEED = base_probe.DEFAULT_SEED
DEFAULT_MATCH_MODE = base_probe.DEFAULT_MATCH_MODE
DEFAULT_MAX_MODEL_LEN = base_probe.DEFAULT_MAX_MODEL_LEN
DEFAULT_NEEDLE_POSITION = base_probe.DEFAULT_NEEDLE_POSITION
NEEDLE_POSITIONS = base_probe.NEEDLE_POSITIONS
NORMALIZATION_RULE = base_probe.NORMALIZATION_RULE
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
    "HTTP 200 from one streaming /v1/chat/completions request",
    "server-sent events are parsed from the streaming response",
    "at least one review-safe streaming delta chunk is received",
    "terminal streaming [DONE] event is received",
    "a final streaming finish_reason is received",
    "narrowly normalized assembled assistant content equals the expected answer",
    "usage token fields are checked when returned and recorded as not_returned otherwise",
    "raw prompt text is not recorded",
    "raw request payload is not recorded",
    "raw generated text is not recorded",
    "raw streaming chunk content is not recorded",
    "token ID arrays are not recorded",
    "logprob values are not recorded",
    "generated-text digests are not recorded",
    "server process group cleanup leaves no remaining PIDs",
]


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _default_stop_sequences(stop_sequences: list[str] | None) -> list[str]:
    if stop_sequences is None:
        return list(DEFAULT_STOP_SEQUENCES)
    return stop_sequences


def _stream_limits(limits: dict[str, Any]) -> dict[str, Any]:
    updated = dict(limits)
    updated["stream"] = True
    updated["assistant_content_recording"] = False
    return updated


def build_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    kwargs["stop_sequences"] = _default_stop_sequences(kwargs.get("stop_sequences"))
    request = base_probe.build_chat_needle_request(**kwargs)
    request["payload"] = dict(request["payload"])
    request["payload"]["stream"] = True
    request["limits"] = _stream_limits(request["limits"])
    return request


def build_planned_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    kwargs["stop_sequences"] = _default_stop_sequences(kwargs.get("stop_sequences"))
    request = base_probe.build_planned_chat_needle_request(**kwargs)
    request["limits"] = _stream_limits(request["limits"])
    return request


def review_safe_request(request: dict[str, Any]) -> dict[str, Any]:
    return base_probe.review_safe_request(request)


def _decode_sse_line(raw_line: Any) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8", errors="replace")
    return str(raw_line)


def _read_sse_events(response: Any) -> list[str]:
    events: list[str] = []
    data_lines: list[str] = []
    while True:
        raw_line = response.readline()
        if raw_line in (b"", ""):
            break
        decoded = _decode_sse_line(raw_line)
        for line in decoded.splitlines():
            if not line:
                if data_lines:
                    events.append("\n".join(data_lines))
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    if data_lines:
        events.append("\n".join(data_lines))
    return events


def _chunk_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    choices = payload.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
    usage = payload.get("usage")
    safe_top_level_keys = {
        "choices",
        "created",
        "id",
        "model",
        "object",
        "service_tier",
        "system_fingerprint",
        "usage",
    }
    safe_choice_keys = {"delta", "finish_reason", "index"}
    safe_delta_keys = {"content", "role"}
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
        "delta_review_safe_keys": sorted(
            key for key in delta.keys() if key in safe_delta_keys
        )
        if isinstance(delta, dict)
        else [],
        "usage_keys": sorted(usage.keys()) if isinstance(usage, dict) else [],
    }


def _review_safe_usage(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        key: usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in usage
    }


def parse_streaming_chat_completion_response(response: Any) -> dict[str, Any]:
    assembled_parts: list[str] = []
    chunk_shapes: list[dict[str, Any]] = []
    event_count = 0
    content_chunk_count = 0
    finish_reason = None
    usage: dict[str, Any] | str = "not_returned"
    model = None
    done_seen = False
    zero_choice_usage_seen = False

    for event in _read_sse_events(response):
        if event == "[DONE]":
            done_seen = True
            continue
        if zero_choice_usage_seen:
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_choice_shape",
                    "usage-only zero-choice chunk must be the final data event",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
                "chunk_shapes": chunk_shapes,
            }
        event_count += 1
        try:
            payload = json.loads(event)
        except json.JSONDecodeError as exc:
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_sse_json",
                    f"streaming SSE data event was not valid JSON: {exc}",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
            }
        chunk_shapes.append(_chunk_shape(payload))
        if not isinstance(payload, dict):
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_chunk_shape",
                    "streaming chunk payload is not a JSON object",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
                "chunk_shapes": chunk_shapes,
            }
        if model is None and isinstance(payload.get("model"), str):
            model = payload["model"]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_choice_shape",
                    "streaming chunk must contain exactly one choice",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
                "chunk_shapes": chunk_shapes,
            }
        if len(choices) == 0 and isinstance(payload.get("usage"), dict):
            usage = _review_safe_usage(payload["usage"])
            zero_choice_usage_seen = True
            continue
        if len(choices) != 1:
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_choice_shape",
                    "streaming chunk must contain exactly one choice",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
                "chunk_shapes": chunk_shapes,
            }
        choice = choices[0]
        if not isinstance(choice, dict):
            return {
                "status": "failed",
                "failure": _failure_payload(
                    "chat_needle_stream_choice_shape",
                    "streaming chunk choice is not an object",
                ),
                "event_count": event_count,
                "content_chunk_count": content_chunk_count,
                "done_seen": done_seen,
                "chunk_shapes": chunk_shapes,
            }
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                assembled_parts.append(content)
                content_chunk_count += 1
        if isinstance(payload.get("usage"), dict):
            usage = _review_safe_usage(payload["usage"])

    return {
        "status": "passed",
        "event_count": event_count,
        "content_chunk_count": content_chunk_count,
        "done_seen": done_seen,
        "finish_reason": finish_reason,
        "model": model,
        "usage": usage,
        "chunk_shapes": chunk_shapes,
        "assembled_content": "".join(assembled_parts),
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


def validate_streaming_chat_needle_response(
    parsed: dict[str, Any],
    *,
    request: dict[str, Any],
    expected_answer: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "expected_answer": expected_answer,
        "match_mode": DEFAULT_MATCH_MODE,
        "normalization": NORMALIZATION_RULE,
        "event_count": parsed.get("event_count", 0),
        "content_chunk_count": parsed.get("content_chunk_count", 0),
        "done_seen": parsed.get("done_seen", False),
        "finish_reason": parsed.get("finish_reason"),
        "chunk_shapes": parsed.get("chunk_shapes", []),
        "checks": {},
    }
    if parsed.get("status") != "passed":
        result["checks"]["stream_parse"] = "failed"
        return _fail_contract(
            result,
            category=parsed.get("failure", {}).get(
                "category",
                "chat_needle_stream_parse_failed",
            ),
            message=parsed.get("failure", {}).get(
                "message",
                "streaming response parsing failed",
            ),
        )
    result["checks"]["stream_parse"] = "passed"

    if parsed.get("event_count", 0) < 1:
        result["checks"]["stream_events_received"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_no_events",
            message="no streaming SSE chunk events were parsed",
        )
    result["checks"]["stream_events_received"] = "passed"

    if parsed.get("content_chunk_count", 0) < 1:
        result["checks"]["stream_content_chunks_received"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_no_content_chunks",
            message="no assistant content delta chunks were parsed",
        )
    result["checks"]["stream_content_chunks_received"] = "passed"

    if not parsed.get("done_seen", False):
        result["checks"]["stream_done_seen"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_done_missing",
            message="terminal streaming [DONE] event was not received",
        )
    result["checks"]["stream_done_seen"] = "passed"

    if parsed.get("finish_reason") is None:
        result["checks"]["stream_finish_reason_present"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_finish_reason_missing",
            message="streaming response did not include a final finish_reason",
        )
    result["checks"]["stream_finish_reason_present"] = "passed"

    usage = parsed.get("usage", "not_returned")
    if usage == "not_returned":
        result["checks"]["usage_shape"] = "not_returned"
        result["usage"] = "not_returned"
    elif isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if not all(
            isinstance(value, int)
            for value in (prompt_tokens, completion_tokens, total_tokens)
        ):
            result["checks"]["usage_shape"] = "failed"
            return _fail_contract(
                result,
                category="chat_needle_stream_usage_shape",
                message=(
                    "usage.prompt_tokens, completion_tokens, and total_tokens "
                    "must be integers when returned"
                ),
            )
        result["checks"]["usage_shape"] = "passed"
        result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        actual_prompt_tokens = request["limits"].get("actual_prompt_tokens")
        if actual_prompt_tokens is None:
            result["checks"]["usage_prompt_tokens_match"] = "not_available"
        elif prompt_tokens != actual_prompt_tokens:
            result["checks"]["usage_prompt_tokens_match"] = "failed"
            return _fail_contract(
                result,
                category="chat_needle_stream_prompt_token_mismatch",
                message=(
                    f"usage.prompt_tokens={prompt_tokens} does not match "
                    f"measured prompt tokens {actual_prompt_tokens}"
                ),
            )
        else:
            result["checks"]["usage_prompt_tokens_match"] = "passed"
        if completion_tokens < 0 or completion_tokens > request["limits"]["max_tokens"]:
            result["checks"]["usage_completion_bound"] = "failed"
            return _fail_contract(
                result,
                category="chat_needle_stream_completion_bound",
                message=(
                    f"usage.completion_tokens={completion_tokens} exceeds "
                    f"request max_tokens={request['limits']['max_tokens']}"
                ),
            )
        result["checks"]["usage_completion_bound"] = "passed"
        if total_tokens < prompt_tokens + completion_tokens:
            result["checks"]["usage_total_tokens"] = "failed"
            return _fail_contract(
                result,
                category="chat_needle_stream_usage_total_tokens",
                message=(
                    "usage.total_tokens must be greater than or equal to "
                    "usage.prompt_tokens + usage.completion_tokens"
                ),
            )
        result["checks"]["usage_total_tokens"] = "passed"
    else:
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_usage_shape",
            message="streaming usage must be an object when returned",
        )

    assembled_content = parsed.get("assembled_content")
    if not isinstance(assembled_content, str):
        result["checks"]["stream_content_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_content_shape",
            message="assembled streaming assistant content is not a string",
        )
    result["checks"]["stream_content_shape"] = "passed"
    normalized_output = base_probe.normalize_generated_output_for_exact_match(
        assembled_content
    )
    result["normalized_output_length_chars"] = len(normalized_output)
    result["normalized_output_equals_expected"] = normalized_output == expected_answer
    if normalized_output != expected_answer:
        result["checks"]["expected_answer_exact"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_stream_expected_answer_not_exact",
            message=(
                "normalized assembled streaming assistant content did not "
                "exactly equal the expected 256K needle answer"
            ),
        )
    result["checks"]["expected_answer_exact"] = "passed"
    return result


def _http_post(url: str, payload: dict[str, Any], timeout: float):
    return base_probe._http_post(url, payload, timeout)


def send_chat_needle_stream_request(
    *,
    port: int,
    request: dict[str, Any],
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post=_http_post,
) -> dict[str, Any]:
    endpoint = request["endpoint"]
    url = f"http://{health_probe.LOCAL_HOST}:{port}{endpoint}"
    result: dict[str, Any] = {
        "status": "failed",
        "endpoint": endpoint,
        "url": url,
        "request_limits": request["limits"],
        "stream_events_received": False,
    }
    try:
        with http_post(url, request["payload"], timeout_seconds) as response:
            status = int(getattr(response, "status", 0))
            result["http_status"] = status
            if status != 200:
                result["failure"] = _failure_payload(
                    "chat_needle_stream_http_status",
                    f"streaming chat completion endpoint returned HTTP {status}",
                )
                return result
            parsed = parse_streaming_chat_completion_response(response)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["failure"] = _failure_payload(
            "chat_needle_stream_http_error",
            _exception_summary(exc),
        )
        return result
    except TimeoutError as exc:
        result["failure"] = _failure_payload(
            "chat_needle_stream_timeout",
            _exception_summary(exc),
        )
        return result
    except Exception as exc:
        result["failure"] = _failure_payload(
            "chat_needle_stream_request_error",
            _exception_summary(exc),
        )
        return result

    result["stream_events_received"] = parsed.get("event_count", 0) > 0
    validation = validate_streaming_chat_needle_response(
        parsed,
        request=request,
        expected_answer=expected_answer,
    )
    result["validation"] = validation
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
                "event_count",
                "content_chunk_count",
                "done_seen",
                "finish_reason",
                "normalized_output_length_chars",
                "normalized_output_equals_expected",
                "usage",
            )
            if key in validation
        }
        result["observation"]["exact_match"] = True
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
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    stop_sequences: list[str] | None = None,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_port = port if port is not None else DEFAULT_PORT
    if selected_port <= 0 or selected_port > 65535:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_port", f"invalid port: {selected_port}"),
            "generation_attempted": False,
            "prompt_sent": False,
            "stream_events_received": False,
            "non_claims": NON_CLAIMS,
        }
    if target_prompt_tokens + max_tokens > max_model_len:
        return {
            "status": "failed",
            "failure": _failure_payload(
                "context_budget_exceeded",
                (
                    f"target_prompt_tokens + max_tokens exceeds max_model_len: "
                    f"{target_prompt_tokens} + {max_tokens} > {max_model_len}"
                ),
            ),
            "generation_attempted": False,
            "prompt_sent": False,
            "stream_events_received": False,
            "non_claims": NON_CLAIMS,
        }
    try:
        request = (
            build_planned_chat_needle_stream_request(
                target_prompt_tokens=target_prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                expected_answer=expected_answer,
                stop_sequences=stop_sequences,
                needle_position=needle_position,
            )
            if dry_run
            else build_chat_needle_stream_request(
                served_model_name=served_model_name,
                artifact_dir=artifact_dir,
                target_prompt_tokens=target_prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                expected_answer=expected_answer,
                stop_sequences=stop_sequences,
                needle_position=needle_position,
            )
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_request", str(exc)),
            "generation_attempted": False,
            "prompt_sent": False,
            "stream_events_received": False,
            "non_claims": NON_CLAIMS,
        }

    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-chat-256k-needle-stream-probe"
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
        "runtime_versions": {} if dry_run else base_probe._runtime_versions(),
        "gpu_memory_before": [] if dry_run else health_probe._query_nvidia_smi_memory(),
        "generation_attempted": False,
        "prompt_sent": False,
        "stream_events_received": False,
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
        readiness_contract = long_prompt_contract.validate_readiness_contract(
            readiness,
            served_model_name=served_model_name,
            max_model_len=max_model_len,
        )
        result["readiness_contract"] = readiness_contract
        if readiness_contract["status"] != "passed":
            result["status"] = "failed"
            result["failure"] = readiness_contract.get(
                "failure",
                _failure_payload("readiness_failed", "server readiness failed"),
            )
            return result
        result["generation_attempted"] = True
        result["prompt_sent"] = True
        completion = send_chat_needle_stream_request(
            port=selected_port,
            request=request,
            expected_answer=expected_answer,
            timeout_seconds=request_timeout_seconds,
        )
        result["stream_completion"] = completion
        result["stream_events_received"] = completion.get(
            "stream_events_received",
            False,
        )
        result["status"] = completion["status"]
        if completion["status"] != "passed":
            result["failure"] = completion.get(
                "failure",
                _failure_payload(
                    "chat_needle_stream_failed",
                    "streaming chat needle exact probe failed",
                ),
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
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=DEFAULT_TARGET_PROMPT_TOKENS,
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-answer", default=DEFAULT_EXPECTED_ANSWER)
    parser.add_argument("--stop-sequence", action="append", dest="stop_sequences")
    parser.add_argument(
        "--needle-position",
        choices=NEEDLE_POSITIONS,
        default=DEFAULT_NEEDLE_POSITION,
    )
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
        target_prompt_tokens=args.target_prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        expected_answer=args.expected_answer,
        stop_sequences=args.stop_sequences,
        needle_position=args.needle_position,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
