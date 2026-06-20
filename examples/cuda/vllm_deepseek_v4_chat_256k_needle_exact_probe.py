#!/usr/bin/env python3
"""Bounded vLLM chat 256K synthetic needle exact probe for DeepSeek V4 Flash."""

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
LONG_PROMPT_CONTRACT_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_long_prompt_response_contract_probe.py"
)
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_TARGET_PROMPT_TOKENS = 255_800
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151"
DEFAULT_MATCH_MODE = "exact"
DEFAULT_MAX_MODEL_LEN = 262_144
DEFAULT_REPEAT_COUNT = 1
DEFAULT_NEEDLE_POSITION = "middle"
NEEDLE_POSITIONS = ("early", "middle", "late")
NORMALIZATION_RULE = (
    "strip leading/trailing whitespace, then strip one surrounding Markdown "
    "code fence when the whole output is fenced"
)
DEFAULT_PROMPT_UNIT = (
    " Synthetic filler for the local-only vLLM chat needle probe; ignore this sentence."
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
    "normalized assistant content equals the expected 256K needle answer",
    "usage prompt/completion/total token fields are internally consistent when returned",
    "usage.prompt_tokens matches measured chat prompt tokens when available",
    "usage.completion_tokens within request max_tokens",
    "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
    "raw prompt text is not recorded",
    "raw request payload is not recorded",
    "raw generated text is not recorded",
    "token ID arrays are not recorded",
    "logprob values are not recorded",
    "generated-text digests are not recorded",
    "server process group cleanup leaves no remaining PIDs",
]

HttpPost = Callable[[str, dict[str, Any], float], Any]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    spec.loader.exec_module(module)
    return module


health_probe = _load_module("vllm_deepseek_v4_server_health_probe", HEALTH_PROBE_PATH)
long_prompt_contract = _load_module(
    "vllm_deepseek_v4_long_prompt_response_contract_probe",
    LONG_PROMPT_CONTRACT_PATH,
)


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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


def _load_tokenizer(artifact_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(artifact_dir),
        trust_remote_code=False,
        local_files_only=True,
    )


def _count_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)
    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_chat_tokens_for_sizing(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[int, bool]:
    try:
        return _count_chat_tokens(tokenizer, messages), True
    except Exception:
        text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return len(tokenizer.encode(text, add_special_tokens=False)), False


def _validate_needle_position(needle_position: str) -> None:
    if needle_position not in NEEDLE_POSITIONS:
        raise ValueError(
            f"needle_position must be one of: {', '.join(NEEDLE_POSITIONS)}"
        )


def _split_filler_units(unit_count: int, needle_position: str) -> tuple[int, int]:
    _validate_needle_position(needle_position)
    if needle_position == "early":
        before_units = unit_count // 10
    elif needle_position == "middle":
        before_units = unit_count // 2
    else:
        before_units = (unit_count * 9) // 10
    before_units = max(0, min(before_units, unit_count))
    after_units = unit_count - before_units
    return before_units, after_units


def _render_user_needle_prompt(
    unit_count: int,
    expected_answer: str,
    *,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
) -> str:
    before_units, after_units = _split_filler_units(unit_count, needle_position)
    prefix = (
        "Bounded local-only vLLM chat synthetic needle retrieval probe.\n"
        "The prompt contains one line labeled NEEDLE_ANSWER. Return only that value.\n"
    )
    needle = f"\nNEEDLE_ANSWER: {expected_answer}\n"
    suffix = (
        "\nQuestion: What is the exact value after NEEDLE_ANSWER? "
        "Reply with only that value.\nAnswer:"
    )
    return (
        prefix
        + DEFAULT_PROMPT_UNIT * before_units
        + needle
        + DEFAULT_PROMPT_UNIT * after_units
        + suffix
    )


def _chat_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return only the exact requested needle string. Do not include "
                "explanation, Markdown, or surrounding text."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _normalize_stop_sequences(stop_sequences: list[str] | None) -> list[str]:
    if stop_sequences is None:
        return []
    normalized = list(stop_sequences)
    if not all(isinstance(value, str) and value for value in normalized):
        raise ValueError("stop sequences must be non-empty strings")
    return normalized


def _validate_request_bounds(
    *,
    target_prompt_tokens: int,
    max_tokens: int,
    expected_answer: str,
    stop_sequences: list[str] | None = None,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
) -> None:
    if target_prompt_tokens < 1:
        raise ValueError("target prompt-token budget must be positive")
    if max_tokens < 1 or max_tokens > DEFAULT_MAX_TOKENS:
        raise ValueError(
            "chat needle requests require "
            f"1 <= max_tokens <= {DEFAULT_MAX_TOKENS}"
        )
    if not expected_answer:
        raise ValueError("expected chat needle answer must be non-empty")
    _validate_needle_position(needle_position)
    _normalize_stop_sequences(stop_sequences)


def _planned_limits(
    *,
    target_prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    expected_answer: str,
    stop_sequences: list[str],
    needle_position: str,
) -> dict[str, Any]:
    limits = {
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": None,
        "tokenizer_accounting": "planned",
        "prompt_chars": None,
        "prompt_unit_chars": len(DEFAULT_PROMPT_UNIT),
        "max_tokens": max_tokens,
        "stream": False,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "expected_answer": expected_answer,
        "match_mode": DEFAULT_MATCH_MODE,
        "needle_position": needle_position,
        "normalization": NORMALIZATION_RULE,
        "stop_sequences_configured": bool(stop_sequences),
        "needle_occurrences": 1,
        "message_count": 2,
        "message_roles": ["system", "user"],
        "prompt_text_recording": False,
        "payload_recording": False,
        "assistant_content_recording": False,
    }
    if stop_sequences:
        limits["stop"] = stop_sequences
    return limits


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


def build_synthetic_chat_needle_messages(
    *,
    artifact_dir: Path,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
) -> dict[str, Any]:
    if target_prompt_tokens < 1:
        raise ValueError("target prompt-token budget must be positive")
    _validate_needle_position(needle_position)
    try:
        tokenizer = _load_tokenizer(artifact_dir)
    except Exception as exc:
        unit_count = max(1, target_prompt_tokens // 8)
        before_units, after_units = _split_filler_units(unit_count, needle_position)
        user_prompt = _render_user_needle_prompt(
            unit_count,
            expected_answer,
            needle_position=needle_position,
        )
        messages = _chat_messages(user_prompt)
        return {
            "messages": messages,
            "accounting": {
                "target_prompt_tokens": target_prompt_tokens,
                "actual_prompt_tokens": None,
                "tokenizer_accounting": "unavailable",
                "tokenizer_error": _exception_summary(exc),
                "prompt_chars": sum(len(message["content"]) for message in messages),
                "prompt_unit_chars": len(DEFAULT_PROMPT_UNIT),
                "needle_occurrences": user_prompt.count(expected_answer),
                "needle_position": needle_position,
                "filler_units_before_needle": before_units,
                "filler_units_after_needle": after_units,
            },
        }

    one_unit_tokens, exact_accounting = _count_chat_tokens_for_sizing(
        tokenizer,
        _chat_messages(_render_user_needle_prompt(1, expected_answer)),
    )
    zero_unit_tokens, zero_exact_accounting = _count_chat_tokens_for_sizing(
        tokenizer,
        _chat_messages(_render_user_needle_prompt(0, expected_answer)),
    )
    exact_accounting = exact_accounting and zero_exact_accounting
    unit_tokens = max(1, one_unit_tokens - zero_unit_tokens)
    low = max(1, target_prompt_tokens // unit_tokens - 128)
    high = max(low, target_prompt_tokens // unit_tokens + 128)
    best_user_prompt = _render_user_needle_prompt(
        low,
        expected_answer,
        needle_position=needle_position,
    )
    best_messages = _chat_messages(best_user_prompt)
    best_unit_count = low
    best_count, best_count_exact = _count_chat_tokens_for_sizing(tokenizer, best_messages)
    exact_accounting = exact_accounting and best_count_exact
    while best_count < target_prompt_tokens and high < target_prompt_tokens * 4:
        candidate_user_prompt = _render_user_needle_prompt(
            high,
            expected_answer,
            needle_position=needle_position,
        )
        candidate_messages = _chat_messages(candidate_user_prompt)
        candidate_count, candidate_exact = _count_chat_tokens_for_sizing(
            tokenizer,
            candidate_messages,
        )
        exact_accounting = exact_accounting and candidate_exact
        if candidate_count >= target_prompt_tokens:
            best_user_prompt = candidate_user_prompt
            best_messages = candidate_messages
            best_unit_count = high
            best_count = candidate_count
            break
        best_user_prompt = candidate_user_prompt
        best_messages = candidate_messages
        best_unit_count = high
        best_count = candidate_count
        low = high
        high *= 2

    left = low
    right = max(left, high)
    while left <= right:
        midpoint = (left + right) // 2
        candidate_user_prompt = _render_user_needle_prompt(
            midpoint,
            expected_answer,
            needle_position=needle_position,
        )
        candidate_messages = _chat_messages(candidate_user_prompt)
        candidate_count, candidate_exact = _count_chat_tokens_for_sizing(
            tokenizer,
            candidate_messages,
        )
        exact_accounting = exact_accounting and candidate_exact
        if abs(candidate_count - target_prompt_tokens) < abs(
            best_count - target_prompt_tokens
        ):
            best_user_prompt = candidate_user_prompt
            best_messages = candidate_messages
            best_unit_count = midpoint
            best_count = candidate_count
        if candidate_count < target_prompt_tokens:
            left = midpoint + 1
        elif candidate_count > target_prompt_tokens:
            right = midpoint - 1
        else:
            best_user_prompt = candidate_user_prompt
            best_messages = candidate_messages
            best_unit_count = midpoint
            best_count = candidate_count
            break

    before_units, after_units = _split_filler_units(best_unit_count, needle_position)
    return {
        "messages": best_messages,
        "accounting": {
            "target_prompt_tokens": target_prompt_tokens,
            "actual_prompt_tokens": best_count if exact_accounting else None,
            "tokenizer_accounting": (
                "transformers.AutoTokenizer chat template"
                if exact_accounting
                else "transformers.AutoTokenizer fallback encode estimate"
            ),
            "prompt_chars": sum(len(message["content"]) for message in best_messages),
            "prompt_unit_chars": len(DEFAULT_PROMPT_UNIT),
            "needle_occurrences": best_user_prompt.count(expected_answer),
            "needle_position": needle_position,
            "filler_units_before_needle": before_units,
            "filler_units_after_needle": after_units,
        },
    }


def build_chat_needle_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    artifact_dir: Path = health_probe.DEFAULT_ARTIFACT_DIR,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    stop_sequences: list[str] | None = None,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
) -> dict[str, Any]:
    _validate_request_bounds(
        target_prompt_tokens=target_prompt_tokens,
        max_tokens=max_tokens,
        expected_answer=expected_answer,
        stop_sequences=stop_sequences,
        needle_position=needle_position,
    )
    normalized_stop_sequences = _normalize_stop_sequences(stop_sequences)
    prompt_result = build_synthetic_chat_needle_messages(
        artifact_dir=artifact_dir,
        target_prompt_tokens=target_prompt_tokens,
        expected_answer=expected_answer,
        needle_position=needle_position,
    )
    if prompt_result["accounting"]["needle_occurrences"] != 1:
        raise ValueError("synthetic chat needle answer must appear exactly once")
    payload = {
        "model": served_model_name,
        "messages": prompt_result["messages"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "n": 1,
        "stream": False,
    }
    if normalized_stop_sequences:
        payload["stop"] = normalized_stop_sequences
    limits = {
        **prompt_result["accounting"],
        "max_tokens": max_tokens,
        "stream": False,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "expected_answer": expected_answer,
        "match_mode": DEFAULT_MATCH_MODE,
        "normalization": NORMALIZATION_RULE,
        "stop_sequences_configured": bool(normalized_stop_sequences),
        "message_count": 2,
        "message_roles": ["system", "user"],
        "prompt_text_recording": False,
        "payload_recording": False,
        "assistant_content_recording": False,
    }
    if normalized_stop_sequences:
        limits["stop"] = normalized_stop_sequences
    return {
        "endpoint": DEFAULT_ENDPOINT,
        "payload": payload,
        "limits": limits,
    }


def build_planned_chat_needle_request(
    *,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    expected_answer: str = DEFAULT_EXPECTED_ANSWER,
    stop_sequences: list[str] | None = None,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
) -> dict[str, Any]:
    _validate_request_bounds(
        target_prompt_tokens=target_prompt_tokens,
        max_tokens=max_tokens,
        expected_answer=expected_answer,
        stop_sequences=stop_sequences,
        needle_position=needle_position,
    )
    normalized_stop_sequences = _normalize_stop_sequences(stop_sequences)
    return {
        "endpoint": DEFAULT_ENDPOINT,
        "limits": _planned_limits(
            target_prompt_tokens=target_prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            expected_answer=expected_answer,
            stop_sequences=normalized_stop_sequences,
            needle_position=needle_position,
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


def validate_chat_needle_response(
    payload: Any,
    *,
    request: dict[str, Any],
    served_model_name: str,
    expected_answer: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "expected_answer": expected_answer,
        "match_mode": DEFAULT_MATCH_MODE,
        "normalization": NORMALIZATION_RULE,
        "response_shape": _response_shape(payload),
        "checks": {},
    }
    if not isinstance(payload, dict):
        result["checks"]["payload_object"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_payload_shape",
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
            category="chat_needle_model_mismatch",
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
            category="chat_needle_choice_shape",
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
            category="chat_needle_choice_fields",
            message=(
                "chat completion choice must include finish_reason and an "
                "assistant message with string content"
            ),
        )
    result["checks"]["choice_fields"] = "passed"
    result["finish_reason"] = choice.get("finish_reason")

    normalized_output = normalize_generated_output_for_exact_match(message["content"])
    result["normalized_output_length_chars"] = len(normalized_output)
    result["normalized_output_equals_expected"] = normalized_output == expected_answer

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_usage_shape",
            message="chat completion response must include a usage object",
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if not all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_usage_shape",
            message="usage.prompt_tokens, completion_tokens, and total_tokens must be integers",
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
            category="chat_needle_prompt_token_mismatch",
            message=(
                f"usage.prompt_tokens={prompt_tokens} does not match measured "
                f"prompt tokens {actual_prompt_tokens}"
            ),
        )
    else:
        result["checks"]["usage_prompt_tokens_match"] = "passed"

    max_tokens = request["limits"]["max_tokens"]
    if completion_tokens < 0 or completion_tokens > max_tokens:
        result["checks"]["usage_completion_bound"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_completion_bound",
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
            category="chat_needle_usage_total_tokens",
            message=(
                "usage.total_tokens must be greater than or equal to "
                "usage.prompt_tokens + usage.completion_tokens"
            ),
        )
    result["checks"]["usage_total_tokens"] = "passed"

    if normalized_output != expected_answer:
        result["checks"]["expected_answer_exact"] = "failed"
        return _fail_contract(
            result,
            category="chat_needle_expected_answer_not_exact",
            message=(
                "normalized assistant content did not exactly equal the "
                "expected 256K needle answer"
            ),
        )
    result["checks"]["expected_answer_exact"] = "passed"
    return result


def send_chat_needle_request(
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
                    "chat_needle_http_status",
                    f"chat completion endpoint returned HTTP {status}",
                )
                return result
            payload = _read_json_response(response)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["failure"] = _failure_payload(
            "chat_needle_http_error",
            _exception_summary(exc),
        )
        return result
    except TimeoutError as exc:
        result["failure"] = _failure_payload(
            "chat_needle_timeout",
            _exception_summary(exc),
        )
        return result
    except Exception as exc:
        result["failure"] = _failure_payload(
            "chat_needle_request_error",
            _exception_summary(exc),
        )
        return result

    validation = validate_chat_needle_response(
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
        result["observation"]["exact_match"] = True
    return result


def repeat_contract_checks(repeat_count: int) -> list[str]:
    if repeat_count == 1:
        return list(CONTRACT_CHECKS)
    checks = []
    for check in CONTRACT_CHECKS:
        if check == "HTTP 200 from one non-streaming /v1/chat/completions request":
            checks.append(
                "HTTP 200 from exactly "
                f"{repeat_count} identical non-streaming "
                "/v1/chat/completions requests in one server lifecycle"
            )
        else:
            checks.append(check)
    checks.append("same review-safe request limits are used for each repeat attempt")
    checks.append("every repeat attempt passes the strict chat needle exact comparator")
    return checks


def review_safe_chat_repeat_attempt_summary(
    completion: dict[str, Any],
    *,
    attempt_index: int,
) -> dict[str, Any]:
    validation = completion.get("validation", {})
    observation = completion.get("observation", {})
    checks = validation.get("checks", {})
    usage = observation.get("usage") or validation.get("usage")
    summary: dict[str, Any] = {
        "attempt_index": attempt_index,
        "status": completion.get("status", "unknown"),
        "endpoint": completion.get("endpoint"),
    }
    if "http_status" in completion:
        summary["http_status"] = completion["http_status"]
    finish_reason = observation.get("finish_reason") or validation.get("finish_reason")
    if finish_reason is not None:
        summary["finish_reason"] = finish_reason
    if "normalized_output_equals_expected" in observation:
        summary["normalized_output_equals_expected"] = observation[
            "normalized_output_equals_expected"
        ]
    elif "normalized_output_equals_expected" in validation:
        summary["normalized_output_equals_expected"] = validation[
            "normalized_output_equals_expected"
        ]
    if "normalized_output_length_chars" in observation:
        summary["normalized_output_length_chars"] = observation[
            "normalized_output_length_chars"
        ]
    elif "normalized_output_length_chars" in validation:
        summary["normalized_output_length_chars"] = validation[
            "normalized_output_length_chars"
        ]
    if "expected_answer_exact" in checks:
        summary["exact_check"] = checks["expected_answer_exact"]
    if isinstance(usage, dict):
        summary["usage"] = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage
        }
    failure = completion.get("failure")
    if isinstance(failure, dict) and "category" in failure:
        summary["failure_category"] = failure["category"]
    return summary


def aggregate_chat_repeat_attempts(
    attempts: list[dict[str, Any]],
    *,
    request: dict[str, Any],
    expected_count: int | None = None,
) -> dict[str, Any]:
    requested_count = expected_count if expected_count is not None else len(attempts)
    summaries = [
        review_safe_chat_repeat_attempt_summary(completion, attempt_index=index)
        for index, completion in enumerate(attempts, start=1)
    ]
    failed_attempts = [
        completion for completion in attempts if completion.get("status") != "passed"
    ]
    incomplete = len(attempts) != requested_count
    aggregate: dict[str, Any] = {
        "status": "failed" if failed_attempts or incomplete else "passed",
        "repeat_count": requested_count,
        "attempts_completed": len(attempts),
        "passed_attempts": len(attempts) - len(failed_attempts),
        "failed_attempts": len(failed_attempts),
        "request": review_safe_request(request),
        "attempts": summaries,
    }
    if incomplete:
        aggregate["failure"] = _failure_payload(
            "chat_needle_repeat_incomplete",
            (
                f"completed {len(attempts)} of {requested_count} requested "
                "repeat attempts"
            ),
        )
    elif failed_attempts:
        aggregate["failure"] = failed_attempts[0].get(
            "failure",
            _failure_payload(
                "chat_needle_repeat_attempt_failed",
                "one or more chat repeat attempts failed",
            ),
        )
    return aggregate


def send_chat_needle_repeat_requests(
    *,
    port: int,
    request: dict[str, Any],
    repeat_count: int,
    served_model_name: str,
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
) -> dict[str, Any]:
    if repeat_count < 1:
        return {
            "status": "failed",
            "failure": _failure_payload(
                "invalid_repeat_count",
                f"repeat_count must be positive, got {repeat_count}",
            ),
        }
    attempts = [
        send_chat_needle_request(
            port=port,
            request=request,
            served_model_name=served_model_name,
            expected_answer=expected_answer,
            timeout_seconds=timeout_seconds,
            http_post=http_post,
        )
        for _ in range(repeat_count)
    ]
    return aggregate_chat_repeat_attempts(
        attempts,
        request=request,
        expected_count=repeat_count,
    )


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
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    needle_position: str = DEFAULT_NEEDLE_POSITION,
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
    if repeat_count < 1:
        return {
            "status": "failed",
            "failure": _failure_payload(
                "invalid_repeat_count",
                f"repeat_count must be positive, got {repeat_count}",
            ),
            "generation_attempted": False,
            "prompt_sent": False,
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
            "non_claims": NON_CLAIMS,
        }
    try:
        request = (
            build_planned_chat_needle_request(
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
            else build_chat_needle_request(
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
            "non_claims": NON_CLAIMS,
        }

    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-chat-256k-needle-exact-probe"
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
        "contract_checks": repeat_contract_checks(repeat_count),
        "repeat_count": repeat_count,
        "request_timeout_seconds": request_timeout_seconds,
        "runtime_versions": {} if dry_run else _runtime_versions(),
        "gpu_memory_before": [] if dry_run else health_probe._query_nvidia_smi_memory(),
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
        if repeat_count == 1:
            completion = send_chat_needle_request(
                port=selected_port,
                request=request,
                served_model_name=served_model_name,
                expected_answer=expected_answer,
                timeout_seconds=request_timeout_seconds,
            )
            result["chat_completion"] = completion
            result["status"] = completion["status"]
            if completion["status"] != "passed":
                result["failure"] = completion.get(
                    "failure",
                    _failure_payload(
                        "chat_needle_failed",
                        "chat needle exact probe failed",
                    ),
                )
        else:
            repeat = send_chat_needle_repeat_requests(
                port=selected_port,
                request=request,
                repeat_count=repeat_count,
                served_model_name=served_model_name,
                expected_answer=expected_answer,
                timeout_seconds=request_timeout_seconds,
            )
            result["repeat"] = repeat
            result["status"] = repeat["status"]
            if repeat["status"] != "passed":
                result["failure"] = repeat.get(
                    "failure",
                    _failure_payload(
                        "chat_needle_repeat_failed",
                        "one or more chat needle repeat attempts failed",
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
        "--repeat-count",
        type=_positive_int,
        default=DEFAULT_REPEAT_COUNT,
        help=(
            "number of repeated chat completion requests to send within one "
            "server lifecycle; defaults to one"
        ),
    )
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
        repeat_count=args.repeat_count,
        needle_position=args.needle_position,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
