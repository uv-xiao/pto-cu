#!/usr/bin/env python3
"""Streaming vLLM chat 256K synthetic needle position-sweep probe."""

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
DEFAULT_PORT = 28_156
DEFAULT_EXPECTED_ANSWER = "PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156"
DEFAULT_NEEDLE_POSITION_SWEEP = ("early", "middle", "late")


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
NEEDLE_POSITIONS = stream_probe.NEEDLE_POSITIONS
NORMALIZATION_RULE = stream_probe.NORMALIZATION_RULE
NON_CLAIMS = stream_probe.NON_CLAIMS

review_safe_request = stream_probe.review_safe_request
send_chat_needle_stream_request = stream_probe.send_chat_needle_stream_request


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _default_stop_sequences(stop_sequences: list[str] | None) -> list[str]:
    return stream_probe._default_stop_sequences(stop_sequences)


def parse_needle_position_sweep(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            "needle-position-sweep requires non-empty comma-separated positions"
        )
    seen: set[str] = set()
    for part in parts:
        if part not in NEEDLE_POSITIONS:
            raise ValueError(
                f"needle_position must be one of: {', '.join(NEEDLE_POSITIONS)}"
            )
        if part in seen:
            raise ValueError(
                f"needle-position-sweep contains duplicate position: {part}"
            )
        seen.add(part)
    return parts


def _parse_needle_position_sweep_arg(value: str) -> list[str]:
    try:
        return parse_needle_position_sweep(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return stream_probe.build_chat_needle_stream_request(**kwargs)


def build_planned_chat_needle_stream_request(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_answer", DEFAULT_EXPECTED_ANSWER)
    return stream_probe.build_planned_chat_needle_stream_request(**kwargs)


def position_sweep_contract_checks(positions: list[str]) -> list[str]:
    checks = []
    for check in stream_probe.CONTRACT_CHECKS:
        if check == "HTTP 200 from one streaming /v1/chat/completions request":
            checks.append(
                "HTTP 200 from one streaming /v1/chat/completions request "
                f"per position in {','.join(positions)} under one server lifecycle"
            )
        else:
            checks.append(check)
    checks.append("every requested position sends exactly one streaming chat request")
    checks.append("every position receives terminal streaming [DONE]")
    checks.append("every position records a final streaming finish_reason")
    checks.append("every position passes the strict chat needle exact comparator")
    return checks


def review_safe_chat_stream_position_summary(
    completion: dict[str, Any],
    *,
    attempt_index: int,
) -> dict[str, Any]:
    validation = completion.get("validation", {})
    observation = completion.get("observation", {})
    checks = validation.get("checks", {})
    request_limits = completion.get("request_limits", {})
    usage = observation.get("usage") or validation.get("usage")
    summary: dict[str, Any] = {
        "attempt_index": attempt_index,
        "position": request_limits.get("needle_position", "unknown"),
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


def aggregate_chat_stream_position_sweep_attempts(
    completions: list[dict[str, Any]],
    *,
    requests: dict[str, dict[str, Any]],
    expected_positions: list[str],
) -> dict[str, Any]:
    summaries = [
        review_safe_chat_stream_position_summary(
            completion,
            attempt_index=index,
        )
        for index, completion in enumerate(completions, start=1)
    ]
    positions_completed = [summary["position"] for summary in summaries]
    seen: set[str] = set()
    duplicate_positions: list[str] = []
    for position in positions_completed:
        if position in seen and position not in duplicate_positions:
            duplicate_positions.append(position)
        seen.add(position)
    missing_positions = [
        position for position in expected_positions if position not in positions_completed
    ]
    failed_positions = [
        completion for completion in completions if completion.get("status") != "passed"
    ]
    aggregate: dict[str, Any] = {
        "status": "failed"
        if duplicate_positions or failed_positions or missing_positions
        else "passed",
        "needle_position_sweep": expected_positions,
        "positions_completed": len(completions),
        "passed_positions": len(completions) - len(failed_positions),
        "failed_positions": len(failed_positions),
        "stream_events_received": all(
            completion.get("stream_events_received", False)
            for completion in completions
        )
        if completions
        else False,
        "requests": [
            review_safe_request(requests[position])
            for position in expected_positions
            if position in requests
        ],
        "positions": summaries,
    }
    if duplicate_positions:
        aggregate["failure"] = _failure_payload(
            "chat_needle_stream_position_sweep_duplicate",
            "position sweep completed duplicate positions: "
            + ", ".join(duplicate_positions),
        )
    elif failed_positions:
        aggregate["failure"] = failed_positions[0].get(
            "failure",
            _failure_payload(
                "chat_needle_stream_position_sweep_position_failed",
                "one or more chat stream position sweep attempts failed",
            ),
        )
    elif missing_positions:
        aggregate["failure"] = _failure_payload(
            "chat_needle_stream_position_sweep_incomplete",
            "position sweep did not complete requested positions: "
            + ", ".join(missing_positions),
        )
    return aggregate


def send_chat_needle_stream_position_sweep_requests(
    *,
    port: int,
    requests: list[dict[str, Any]],
    expected_answer: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post=stream_probe._http_post,
) -> dict[str, Any]:
    requests_by_position = {
        request["limits"].get("needle_position", "unknown"): request
        for request in requests
    }
    expected_positions = [
        request["limits"].get("needle_position", "unknown")
        for request in requests
    ]
    completions = [
        send_chat_needle_stream_request(
            port=port,
            request=request,
            expected_answer=expected_answer,
            timeout_seconds=timeout_seconds,
            http_post=http_post,
        )
        for request in requests
    ]
    return aggregate_chat_stream_position_sweep_attempts(
        completions,
        requests=requests_by_position,
        expected_positions=expected_positions,
    )


def _build_requests(
    *,
    dry_run: bool,
    positions: list[str],
    served_model_name: str,
    artifact_dir: Path,
    target_prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    expected_answer: str,
    stop_sequences: list[str] | None,
) -> dict[str, dict[str, Any]]:
    builder = (
        build_planned_chat_needle_stream_request
        if dry_run
        else build_chat_needle_stream_request
    )
    requests: dict[str, dict[str, Any]] = {}
    for position in positions:
        kwargs: dict[str, Any] = {
            "target_prompt_tokens": target_prompt_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "expected_answer": expected_answer,
            "stop_sequences": _default_stop_sequences(stop_sequences),
            "needle_position": position,
        }
        if not dry_run:
            kwargs["served_model_name"] = served_model_name
            kwargs["artifact_dir"] = artifact_dir
        requests[position] = builder(**kwargs)
    return requests


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
    needle_position_sweep: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_port = port if port is not None else DEFAULT_PORT
    positions = list(needle_position_sweep or DEFAULT_NEEDLE_POSITION_SWEEP)
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
        parse_needle_position_sweep(",".join(positions))
        requests = _build_requests(
            dry_run=dry_run,
            positions=positions,
            served_model_name=served_model_name,
            artifact_dir=artifact_dir,
            target_prompt_tokens=target_prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            expected_answer=expected_answer,
            stop_sequences=stop_sequences,
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

    first_request = requests[positions[0]]
    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-chat-256k-needle-stream-position-sweep-probe"
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
        "request": review_safe_request(first_request),
        "contract_checks": position_sweep_contract_checks(positions),
        "needle_position_sweep": positions,
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
        completions = []
        for position in positions:
            completions.append(
                send_chat_needle_stream_request(
                    port=selected_port,
                    request=requests[position],
                    expected_answer=expected_answer,
                    timeout_seconds=request_timeout_seconds,
                )
            )
            if process.poll() is not None or completions[-1].get("status") != "passed":
                break
        sweep = aggregate_chat_stream_position_sweep_attempts(
            completions,
            requests=requests,
            expected_positions=positions,
        )
        result["sweep"] = sweep
        result["stream_events_received"] = sweep.get("stream_events_received", False)
        result["status"] = sweep["status"]
        if sweep["status"] != "passed":
            result["failure"] = sweep.get(
                "failure",
                _failure_payload(
                    "chat_needle_stream_position_sweep_failed",
                    "one or more chat stream position sweep attempts failed",
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
        "--needle-position-sweep",
        type=_parse_needle_position_sweep_arg,
        default=list(DEFAULT_NEEDLE_POSITION_SWEEP),
        help=(
            "comma-separated unique needle positions to request under one "
            "server lifecycle"
        ),
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
        needle_position_sweep=args.needle_position_sweep,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
