#!/usr/bin/env python3
"""Two-request streaming vLLM chat 256K synthetic needle exact-repeat probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAM_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_chat_256k_needle_stream_probe.py"
)
DEFAULT_PORT = 28_154
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_STREAM_REPEAT_OK_28154"
DEFAULT_REPEAT_COUNT = 2


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
base_probe = stream_probe.base_probe
health_probe = stream_probe.health_probe
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
NON_CLAIMS = stream_probe.NON_CLAIMS

review_safe_request = stream_probe.review_safe_request
send_chat_needle_stream_request = stream_probe.send_chat_needle_stream_request


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _default_stop_sequences(stop_sequences: list[str] | None) -> list[str]:
    return stream_probe._default_stop_sequences(stop_sequences)


def build_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return stream_probe.build_chat_needle_stream_request(**kwargs)


def build_planned_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return stream_probe.build_planned_chat_needle_stream_request(**kwargs)


def repeat_contract_checks(repeat_count: int) -> list[str]:
    if repeat_count == 1:
        return list(stream_probe.CONTRACT_CHECKS)
    checks = []
    for check in stream_probe.CONTRACT_CHECKS:
        if check == "HTTP 200 from one streaming /v1/chat/completions request":
            checks.append(
                "HTTP 200 from exactly "
                f"{repeat_count} identical streaming "
                "/v1/chat/completions requests in one server lifecycle"
            )
        else:
            checks.append(check)
    checks.append("the same streaming request payload is posted for each repeat attempt")
    checks.append("every repeat attempt receives terminal streaming [DONE]")
    checks.append("every repeat attempt records a final streaming finish_reason")
    checks.append("every repeat attempt passes the strict chat needle exact comparator")
    return checks


def review_safe_chat_stream_repeat_attempt_summary(
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
        "stream_events_received": completion.get("stream_events_received", False),
    }
    if "http_status" in completion:
        summary["http_status"] = completion["http_status"]
    for source in (observation, validation):
        if "event_count" in source:
            summary["event_count"] = source["event_count"]
        if "content_chunk_count" in source:
            summary["content_chunk_count"] = source["content_chunk_count"]
        if "done_seen" in source:
            summary["stream_done_seen"] = source["done_seen"]
        if source.get("finish_reason") is not None:
            summary["finish_reason"] = source["finish_reason"]
        if "normalized_output_equals_expected" in source:
            summary["normalized_output_equals_expected"] = source[
                "normalized_output_equals_expected"
            ]
        if "normalized_output_length_chars" in source:
            summary["normalized_output_length_chars"] = source[
                "normalized_output_length_chars"
            ]
    summary["stream_finish_reason_present"] = checks.get(
        "stream_finish_reason_present"
    )
    summary["exact_check"] = checks.get("expected_answer_exact")
    if isinstance(usage, dict):
        summary["usage"] = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage
        }
    elif usage == "not_returned":
        summary["usage"] = "not_returned"
    failure = completion.get("failure")
    if isinstance(failure, dict) and "category" in failure:
        summary["failure_category"] = failure["category"]
    return summary


def aggregate_chat_stream_repeat_attempts(
    attempts: list[dict[str, Any]],
    *,
    request: dict[str, Any],
    expected_count: int | None = None,
) -> dict[str, Any]:
    requested_count = expected_count if expected_count is not None else len(attempts)
    summaries = [
        review_safe_chat_stream_repeat_attempt_summary(
            completion,
            attempt_index=index,
        )
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
        "stream_events_received": all(
            completion.get("stream_events_received", False) for completion in attempts
        )
        if attempts
        else False,
        "request": review_safe_request(request),
        "attempts": summaries,
    }
    if incomplete:
        aggregate["failure"] = _failure_payload(
            "chat_needle_stream_repeat_incomplete",
            (
                f"completed {len(attempts)} of {requested_count} requested "
                "stream repeat attempts"
            ),
        )
    elif failed_attempts:
        aggregate["failure"] = failed_attempts[0].get(
            "failure",
            _failure_payload(
                "chat_needle_stream_repeat_attempt_failed",
                "one or more chat stream repeat attempts failed",
            ),
        )
    return aggregate


def send_chat_needle_stream_repeat_requests(
    *,
    port: int,
    request: dict[str, Any],
    repeat_count: int,
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post=stream_probe._http_post,
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
        send_chat_needle_stream_request(
            port=port,
            request=request,
            expected_answer=expected_answer,
            timeout_seconds=timeout_seconds,
            http_post=http_post,
        )
        for _ in range(repeat_count)
    ]
    return aggregate_chat_stream_repeat_attempts(
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
    if repeat_count < 1:
        return {
            "status": "failed",
            "failure": _failure_payload(
                "invalid_repeat_count",
                f"repeat_count must be positive, got {repeat_count}",
            ),
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
                stop_sequences=_default_stop_sequences(stop_sequences),
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
                stop_sequences=_default_stop_sequences(stop_sequences),
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
        / "vllm-chat-256k-needle-stream-repeat-probe"
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
        repeat = send_chat_needle_stream_repeat_requests(
            port=selected_port,
            request=request,
            repeat_count=repeat_count,
            expected_answer=expected_answer,
            timeout_seconds=request_timeout_seconds,
        )
        result["repeat"] = repeat
        result["stream_events_received"] = repeat.get("stream_events_received", False)
        result["status"] = repeat["status"]
        if repeat["status"] != "passed":
            result["failure"] = repeat.get(
                "failure",
                _failure_payload(
                    "chat_needle_stream_repeat_failed",
                    "one or more chat stream repeat attempts failed",
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
        "--repeat-count",
        type=base_probe._positive_int,
        default=DEFAULT_REPEAT_COUNT,
        help=(
            "number of repeated streaming chat completion requests to send "
            "within one server lifecycle"
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
