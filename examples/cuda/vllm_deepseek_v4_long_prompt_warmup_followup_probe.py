#!/usr/bin/env python3
"""Bounded vLLM long-prompt warmup/follow-up probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESPONSE_CONTRACT_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_long_prompt_response_contract_probe.py"
)
DEFAULT_LOG_SETTLE_SECONDS = 2.0
NON_CLAIMS = [
    "not generated-text correctness evidence",
    "not tokenizer semantic correctness evidence",
    "not prompt semantic correctness evidence",
    "not token identity evidence",
    "not logprob evidence",
    "not stop-token evidence",
    "not throughput or latency evidence",
    "not production-readiness evidence",
    "not broad determinism evidence",
    "not simpler-nv or vLLM integration evidence",
]
CONTRACT_CHECKS = [
    "HTTP 200 from /health",
    "HTTP 200 from /v1/models",
    "model list includes served model and max_model_len=262144",
    "HTTP 200 from warmup non-streaming /v1/completions request",
    "HTTP 200 from followup non-streaming /v1/completions request",
    "top-level completion responses are JSON objects",
    "response model fields match served model when returned",
    "exactly one response choice object per response",
    "first choices expose text and finish_reason fields",
    "generated text lengths are recorded without generated text contents",
    "usage prompt/completion/total token fields are internally consistent when returned",
    "usage.prompt_tokens matches measured prompt tokens when available",
    "usage.completion_tokens within request max_tokens",
    "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
    "raw prompt text is not recorded",
    "raw generated text is not recorded",
    "server process group cleanup leaves no remaining PIDs",
]


def _load_response_contract_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_long_prompt_response_contract_probe",
        RESPONSE_CONTRACT_PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {RESPONSE_CONTRACT_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


response_contract_probe = _load_response_contract_probe()
health_probe = response_contract_probe.health_probe


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _review_safe_request(label: str, request: dict[str, Any]) -> dict[str, Any]:
    safe = response_contract_probe._review_safe_request(request)
    return {
        "label": label,
        **safe,
    }


def _server_log_snippets(log_path: Path, *, max_lines: int = 80) -> list[str]:
    return response_contract_probe._server_log_snippets(
        log_path,
        max_lines=max_lines,
    )


def _failure_note(
    result: dict[str, Any],
    *,
    max_model_len: int,
    target_prompt_tokens: int,
    selected_port: int,
    log_path: Path,
) -> dict[str, Any]:
    failure = result.get("failure", {})
    return {
        "boundary": {
            "server_host": health_probe.LOCAL_HOST,
            "server_port": selected_port,
            "max_model_len": max_model_len,
            "target_prompt_tokens": target_prompt_tokens,
            "max_tokens": response_contract_probe.DEFAULT_MAX_TOKENS,
            "request_shape": "two consecutive same-shape non-streaming completions",
        },
        "failed_request": result.get("failed_request", "unknown"),
        "failure_category": failure.get("category", "unknown"),
        "failure_message": failure.get("message", ""),
        "server_log_tail": _server_log_snippets(log_path),
        "cleanup": result.get("cleanup", {}),
        "next_diagnostic_gate": (
            "rerun the same long-prompt warmup/follow-up probe after checking "
            "server readiness, model-list metadata, and per-request usage accounting"
        ),
    }


def run_probe(
    *,
    artifact_dir: Path = health_probe.DEFAULT_ARTIFACT_DIR,
    vllm_bin: Path = health_probe.DEFAULT_VLLM_BIN,
    port: int | None = None,
    server_log: Path | None = None,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    max_model_len: int = 262_144,
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
    request_timeout_seconds: float = response_contract_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    terminate_timeout_seconds: float = health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    target_prompt_tokens: int = response_contract_probe.DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = response_contract_probe.DEFAULT_MAX_TOKENS,
    temperature: float = response_contract_probe.DEFAULT_TEMPERATURE,
    top_p: float = response_contract_probe.DEFAULT_TOP_P,
    seed: int = response_contract_probe.DEFAULT_SEED,
    log_settle_seconds: float = DEFAULT_LOG_SETTLE_SECONDS,
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
        if dry_run:
            contract_request = (
                response_contract_probe.build_planned_long_prompt_contract_request(
                    target_prompt_tokens=target_prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            )
        else:
            contract_request = response_contract_probe.build_long_prompt_contract_request(
                served_model_name=served_model_name,
                artifact_dir=artifact_dir,
                target_prompt_tokens=target_prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
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
        / "vllm-long-prompt-warmup-followup-probe"
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
        "endpoints": [
            "/health",
            "/v1/models",
            response_contract_probe.DEFAULT_ENDPOINT,
            response_contract_probe.DEFAULT_ENDPOINT,
        ],
        "warmup_request": _review_safe_request("warmup", contract_request),
        "followup_request": _review_safe_request("followup", contract_request),
        "contract_checks": CONTRACT_CHECKS,
        "request_timeout_seconds": request_timeout_seconds,
        "log_settle_seconds": log_settle_seconds,
        "runtime_versions": {}
        if dry_run
        else health_probe._runtime_versions(),
        "gpu_memory_before": []
        if dry_run
        else health_probe._query_nvidia_smi_memory(),
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
        readiness_contract = response_contract_probe.validate_readiness_contract(
            readiness,
            served_model_name=served_model_name,
            max_model_len=max_model_len,
        )
        result["readiness_contract"] = readiness_contract
        if readiness_contract["status"] != "passed":
            result["status"] = "failed"
            result["failed_request"] = "readiness"
            result["failure"] = readiness_contract.get(
                "failure",
                _failure_payload("readiness_failed", "server readiness failed"),
            )
            return result

        for label, key in [
            ("warmup", "warmup_completion"),
            ("followup", "followup_completion"),
        ]:
            completion = response_contract_probe.send_contract_request(
                port=selected_port,
                request=contract_request,
                served_model_name=served_model_name,
                timeout_seconds=request_timeout_seconds,
            )
            result["generation_attempted"] = True
            result["prompt_sent"] = True
            result[key] = completion
            time.sleep(log_settle_seconds)
            if completion["status"] != "passed":
                result["status"] = "failed"
                result["failed_request"] = label
                result["failure"] = completion.get(
                    "failure",
                    _failure_payload(
                        f"{label}_long_prompt_completion_failed",
                        f"{label} long-prompt completion failed",
                    ),
                )
                return result

        result["status"] = "passed"
        if process.poll() is not None:
            result["status"] = "failed"
            result["failed_request"] = "server_lifecycle"
            result["failure"] = _failure_payload(
                "server_exited_during_probe",
                f"server process exited with {process.returncode}",
            )
    except Exception as exc:
        result["status"] = "failed"
        result["failed_request"] = "probe"
        result["failure"] = _failure_payload("probe_error", _exception_summary(exc))
    finally:
        if process is not None:
            result["cleanup"] = health_probe.cleanup_process(
                process,
                terminate_timeout_seconds=terminate_timeout_seconds,
            )
            if result["cleanup"]["status"] != "passed":
                result["status"] = "failed"
                result["failed_request"] = "cleanup"
                result["failure"] = _failure_payload(
                    "cleanup_failed",
                    "server process group still had remaining processes after cleanup",
                )
        if log_file is not None:
            log_file.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["gpu_memory_after"] = health_probe._query_nvidia_smi_memory()
        if result["status"] != "passed":
            result["failure_note"] = _failure_note(
                result,
                max_model_len=max_model_len,
                target_prompt_tokens=target_prompt_tokens,
                selected_port=selected_port,
                log_path=log_path,
            )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=health_probe.DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vllm-bin", type=Path, default=health_probe.DEFAULT_VLLM_BIN)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--served-model-name", default=health_probe.DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=262_144)
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
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=health_probe.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=health_probe.DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=response_contract_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--terminate-timeout-seconds",
        type=float,
        default=health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=response_contract_probe.DEFAULT_TARGET_PROMPT_TOKENS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=response_contract_probe.DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=response_contract_probe.DEFAULT_TEMPERATURE,
    )
    parser.add_argument("--top-p", type=float, default=response_contract_probe.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=response_contract_probe.DEFAULT_SEED)
    parser.add_argument("--log-settle-seconds", type=float, default=DEFAULT_LOG_SETTLE_SECONDS)
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
        log_settle_seconds=args.log_settle_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
