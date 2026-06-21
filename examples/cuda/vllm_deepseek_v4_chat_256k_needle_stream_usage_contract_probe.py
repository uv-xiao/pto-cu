#!/usr/bin/env python3
"""Streaming vLLM chat 256K usage contract probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAM_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_stream_probe.py"
)
DEFAULT_PORT = 28_157
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28157"


def _load_stream_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_chat_256k_needle_stream_probe",
        STREAM_PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {STREAM_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


stream_probe = _load_stream_probe()

health_probe = stream_probe.health_probe
base_probe = stream_probe.base_probe
long_prompt_contract = stream_probe.long_prompt_contract

DEFAULT_ENDPOINT = stream_probe.DEFAULT_ENDPOINT
DEFAULT_TARGET_PROMPT_TOKENS = stream_probe.DEFAULT_TARGET_PROMPT_TOKENS
DEFAULT_REQUEST_TIMEOUT_SECONDS = stream_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS
DEFAULT_MAX_TOKENS = stream_probe.DEFAULT_MAX_TOKENS
DEFAULT_TEMPERATURE = stream_probe.DEFAULT_TEMPERATURE
DEFAULT_TOP_P = stream_probe.DEFAULT_TOP_P
DEFAULT_SEED = stream_probe.DEFAULT_SEED
DEFAULT_MATCH_MODE = stream_probe.DEFAULT_MATCH_MODE
DEFAULT_MAX_MODEL_LEN = stream_probe.DEFAULT_MAX_MODEL_LEN
DEFAULT_NEEDLE_POSITION = stream_probe.DEFAULT_NEEDLE_POSITION
NEEDLE_POSITIONS = stream_probe.NEEDLE_POSITIONS
NORMALIZATION_RULE = stream_probe.NORMALIZATION_RULE

NON_CLAIMS = stream_probe.NON_CLAIMS
CONTRACT_CHECKS = [
    *stream_probe.CONTRACT_CHECKS,
    "stream_options.include_usage=true is sent with the request",
    "streaming usage object is returned",
    "usage.prompt_tokens matches measured prompt tokens",
    "usage.completion_tokens is within max_tokens",
    "usage.total_tokens is at least prompt_tokens + completion_tokens",
]

parse_streaming_chat_completion_response = (
    stream_probe.parse_streaming_chat_completion_response
)

_raw_build_chat_needle_stream_request = stream_probe.build_chat_needle_stream_request
_raw_build_planned_chat_needle_stream_request = (
    stream_probe.build_planned_chat_needle_stream_request
)
_raw_send_chat_needle_stream_request = stream_probe.send_chat_needle_stream_request


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _with_usage_contract(request: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request)
    if "payload" in request:
        updated["payload"] = dict(request["payload"])
        updated["payload"]["stream_options"] = {"include_usage": True}
    updated["limits"] = dict(request["limits"])
    updated["limits"]["stream"] = True
    updated["limits"]["stream_options_include_usage"] = True
    return updated


def build_chat_needle_stream_usage_contract_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return _with_usage_contract(_raw_build_chat_needle_stream_request(**kwargs))


def build_planned_chat_needle_stream_usage_contract_request(
    **kwargs: Any,
) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return _with_usage_contract(_raw_build_planned_chat_needle_stream_request(**kwargs))


def review_safe_request(request: dict[str, Any]) -> dict[str, Any]:
    return stream_probe.review_safe_request(request)


def _fail_usage_contract(
    result: dict[str, Any],
    validation: dict[str, Any],
    *,
    category: str,
    message: str,
) -> dict[str, Any]:
    result["status"] = "failed"
    validation["status"] = "failed"
    validation["failure"] = _failure_payload(category, message)
    result["failure"] = validation["failure"]
    return result


def _enforce_usage_contract(result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("validation")
    if not isinstance(validation, dict):
        return result
    checks = validation.setdefault("checks", {})
    if validation.get("status") != "passed":
        return result

    usage = validation.get("usage", "not_returned")
    if usage == "not_returned":
        checks["usage_presence"] = "failed"
        return _fail_usage_contract(
            result,
            validation,
            category="chat_needle_stream_usage_not_returned",
            message="streaming usage was not returned despite include_usage=true",
        )
    if not isinstance(usage, dict):
        checks["usage_presence"] = "failed"
        return _fail_usage_contract(
            result,
            validation,
            category="chat_needle_stream_usage_shape",
            message="streaming usage must be an object",
        )
    checks["usage_presence"] = "passed"

    required_passed_checks = {
        "usage_shape": "chat_needle_stream_usage_shape",
        "usage_prompt_tokens_match": "chat_needle_stream_prompt_token_mismatch",
        "usage_completion_bound": "chat_needle_stream_completion_bound",
        "usage_total_tokens": "chat_needle_stream_usage_total_tokens",
    }
    for check_name, category in required_passed_checks.items():
        if checks.get(check_name) != "passed":
            return _fail_usage_contract(
                result,
                validation,
                category=category,
                message=f"{check_name} did not pass",
            )

    observation = result.setdefault("observation", {})
    observation["usage"] = usage
    observation["usage_shape"] = checks["usage_shape"]
    observation["usage_prompt_tokens_match"] = checks["usage_prompt_tokens_match"]
    observation["usage_completion_bound"] = checks["usage_completion_bound"]
    observation["usage_total_tokens"] = checks["usage_total_tokens"]
    return result


def send_chat_needle_stream_usage_contract_request(
    *,
    port: int,
    request: dict[str, Any],
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post=stream_probe._http_post,
) -> dict[str, Any]:
    request = _with_usage_contract(request)
    result = _raw_send_chat_needle_stream_request(
        port=port,
        request=request,
        expected_answer=expected_answer,
        timeout_seconds=timeout_seconds,
        http_post=http_post,
    )
    return _enforce_usage_contract(result)


def run_probe(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("port", DEFAULT_PORT)
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    original_build = stream_probe.build_chat_needle_stream_request
    original_planned = stream_probe.build_planned_chat_needle_stream_request
    original_send = stream_probe.send_chat_needle_stream_request
    try:
        stream_probe.build_chat_needle_stream_request = (
            build_chat_needle_stream_usage_contract_request
        )
        stream_probe.build_planned_chat_needle_stream_request = (
            build_planned_chat_needle_stream_usage_contract_request
        )
        stream_probe.send_chat_needle_stream_request = (
            send_chat_needle_stream_usage_contract_request
        )
        result = stream_probe.run_probe(**kwargs)
    finally:
        stream_probe.build_chat_needle_stream_request = original_build
        stream_probe.build_planned_chat_needle_stream_request = original_planned
        stream_probe.send_chat_needle_stream_request = original_send
    result["contract_checks"] = CONTRACT_CHECKS
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
