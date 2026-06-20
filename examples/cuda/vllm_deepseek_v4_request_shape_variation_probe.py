#!/usr/bin/env python3
"""Bounded vLLM request-shape variation probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WARMUP_SHAPE_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_warmup_shape_probe.py"
)
DEFAULT_VARIATION_PROMPT = (
    "Hello. Provide one short sentence about bounded CUDA serving probes."
)
DEFAULT_VARIATION_MAX_TOKENS = 8
NON_CLAIMS = [
    "not generated-text correctness evidence",
    "not tokenizer semantic correctness evidence",
    "not prompt correctness evidence",
    "not 256K context evidence",
    "not throughput or latency evidence",
    "not production-readiness evidence",
    "not broad warmup-eliminates-JIT evidence",
    "not simpler-nv or vLLM kernel integration evidence",
]


def _load_warmup_shape_probe():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_warmup_shape_probe",
        WARMUP_SHAPE_PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"unable to load {WARMUP_SHAPE_PROBE_PATH}")
    spec.loader.exec_module(module)
    return module


warmup_shape_probe = _load_warmup_shape_probe()
response_contract_probe = warmup_shape_probe.response_contract_probe
health_probe = warmup_shape_probe.health_probe


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _log_size(path: Path) -> int:
    return warmup_shape_probe._log_size(path)


def _read_log_slice(path: Path, start: int, end: int | None = None) -> str:
    return warmup_shape_probe._read_log_slice(path, start, end)


def summarize_jit_warning_windows(
    *,
    log_path: Path,
    offsets: dict[str, int],
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    selected_patterns = patterns or warmup_shape_probe.DEFAULT_JIT_WARNING_PATTERNS
    windows = {
        "before_warmup": (0, offsets["before_warmup"]),
        "warmup_request": (offsets["before_warmup"], offsets["after_warmup"]),
        "followup_same_shape_request": (
            offsets["after_warmup"],
            offsets["after_followup"],
        ),
        "variation_request": (offsets["after_followup"], offsets["after_variation"]),
        "full_before_cleanup": (0, offsets["after_variation"]),
    }
    return {
        "patterns": selected_patterns,
        "window_mode": "server log byte offsets captured around synchronous requests",
        "windows": {
            name: {
                "byte_start": start,
                "byte_end": end,
                **warmup_shape_probe.summarize_jit_warnings(
                    _read_log_slice(log_path, start, end),
                    patterns=selected_patterns,
                ),
            }
            for name, (start, end) in windows.items()
        },
    }


def _build_contract_request_or_failure(
    *,
    served_model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    failure_category: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        request = response_contract_probe.build_contract_request(
            served_model_name=served_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    except ValueError as exc:
        return None, {
            "status": "failed",
            "failure": _failure_payload(failure_category, str(exc)),
            "generation_attempted": False,
            "non_claims": NON_CLAIMS,
        }
    return request, None


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
    request_timeout_seconds: float = response_contract_probe.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    terminate_timeout_seconds: float = health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    prompt: str = response_contract_probe.DEFAULT_PROMPT,
    max_tokens: int = response_contract_probe.DEFAULT_MAX_TOKENS,
    variation_prompt: str = DEFAULT_VARIATION_PROMPT,
    variation_max_tokens: int = DEFAULT_VARIATION_MAX_TOKENS,
    temperature: float = response_contract_probe.DEFAULT_TEMPERATURE,
    top_p: float = response_contract_probe.DEFAULT_TOP_P,
    seed: int = response_contract_probe.DEFAULT_SEED,
    same_shape_followup: bool = True,
    log_settle_seconds: float = warmup_shape_probe.DEFAULT_LOG_SETTLE_SECONDS,
    jit_warning_patterns: list[str] | None = None,
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
    warmup_request, failure = _build_contract_request_or_failure(
        served_model_name=served_model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        failure_category="invalid_warmup_request",
    )
    if failure is not None:
        return failure
    variation_request, failure = _build_contract_request_or_failure(
        served_model_name=served_model_name,
        prompt=variation_prompt,
        max_tokens=variation_max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        failure_category="invalid_variation_request",
    )
    if failure is not None:
        return failure
    assert warmup_request is not None
    assert variation_request is not None

    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-request-shape-variation-probe"
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
    endpoints = [
        "/health",
        "/v1/models",
        response_contract_probe.DEFAULT_ENDPOINT,
    ]
    if same_shape_followup:
        endpoints.append(response_contract_probe.DEFAULT_ENDPOINT)
    endpoints.append(response_contract_probe.DEFAULT_ENDPOINT)

    result: dict[str, Any] = {
        "status": "planned" if dry_run else "failed",
        "server_command": command,
        "server_host": health_probe.LOCAL_HOST,
        "server_port": selected_port,
        "server_log": health_probe._display_path(log_path),
        "endpoints": endpoints,
        "warmup_request": {
            "label": "warmup",
            **warmup_request,
        },
        "variation_request": {
            "label": "variation",
            **variation_request,
        },
        "contract_checks": [
            "HTTP 200 from /health",
            "HTTP 200 from /v1/models",
            "HTTP 200 from warmup /v1/completions",
            "HTTP 200 from variation /v1/completions",
            "response-contract checks pass for every completion response",
            "server log inspected for selected JIT warning strings by request window",
            "server process group cleanup leaves no remaining PIDs",
        ],
        "request_timeout_seconds": request_timeout_seconds,
        "log_settle_seconds": log_settle_seconds,
        "jit_warning_patterns": (
            jit_warning_patterns or warmup_shape_probe.DEFAULT_JIT_WARNING_PATTERNS
        ),
        "runtime_versions": health_probe._runtime_versions(),
        "gpu_memory_before": health_probe._query_nvidia_smi_memory(),
        "generation_attempted": False,
        "non_claims": NON_CLAIMS,
    }
    if same_shape_followup:
        result["followup_same_shape_request"] = {
            "label": "followup_same_shape",
            **warmup_request,
        }
    if dry_run:
        return result

    started = time.monotonic()
    process = None
    log_file = None
    offsets: dict[str, int] = {}
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

        offsets["before_warmup"] = _log_size(log_path)
        warmup = response_contract_probe.send_contract_request(
            port=selected_port,
            request=warmup_request,
            timeout_seconds=request_timeout_seconds,
        )
        result["generation_attempted"] = True
        result["warmup_completion"] = warmup
        time.sleep(log_settle_seconds)
        offsets["after_warmup"] = _log_size(log_path)
        if warmup["status"] != "passed":
            result["status"] = "failed"
            result["failure"] = warmup.get(
                "failure",
                _failure_payload("warmup_completion_failed", "warmup request failed"),
            )
            return result

        if same_shape_followup:
            followup = response_contract_probe.send_contract_request(
                port=selected_port,
                request=warmup_request,
                timeout_seconds=request_timeout_seconds,
            )
            result["followup_same_shape_completion"] = followup
            time.sleep(log_settle_seconds)
            offsets["after_followup"] = _log_size(log_path)
            if followup["status"] != "passed":
                result["status"] = "failed"
                result["failure"] = followup.get(
                    "failure",
                    _failure_payload(
                        "followup_same_shape_completion_failed",
                        "follow-up same-shape request failed",
                    ),
                )
                return result
        else:
            offsets["after_followup"] = offsets["after_warmup"]

        variation = response_contract_probe.send_contract_request(
            port=selected_port,
            request=variation_request,
            timeout_seconds=request_timeout_seconds,
        )
        result["variation_completion"] = variation
        time.sleep(log_settle_seconds)
        offsets["after_variation"] = _log_size(log_path)
        result["jit_warning_summary"] = summarize_jit_warning_windows(
            log_path=log_path,
            offsets=offsets,
            patterns=jit_warning_patterns,
        )
        result["status"] = variation["status"]
        if variation["status"] != "passed":
            result["failure"] = variation.get(
                "failure",
                _failure_payload(
                    "variation_completion_failed",
                    "variation request failed",
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
        if offsets and "jit_warning_summary" not in result:
            offsets.setdefault("after_warmup", _log_size(log_path))
            offsets.setdefault("after_followup", offsets["after_warmup"])
            offsets.setdefault("after_variation", offsets["after_followup"])
            result["jit_warning_summary"] = summarize_jit_warning_windows(
                log_path=log_path,
                offsets=offsets,
                patterns=jit_warning_patterns,
            )
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
    parser.add_argument("--prompt", default=response_contract_probe.DEFAULT_PROMPT)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=response_contract_probe.DEFAULT_MAX_TOKENS,
    )
    parser.add_argument("--variation-prompt", default=DEFAULT_VARIATION_PROMPT)
    parser.add_argument(
        "--variation-max-tokens",
        type=int,
        default=DEFAULT_VARIATION_MAX_TOKENS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=response_contract_probe.DEFAULT_TEMPERATURE,
    )
    parser.add_argument("--top-p", type=float, default=response_contract_probe.DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=response_contract_probe.DEFAULT_SEED)
    parser.add_argument(
        "--same-shape-followup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-settle-seconds",
        type=float,
        default=warmup_shape_probe.DEFAULT_LOG_SETTLE_SECONDS,
    )
    parser.add_argument(
        "--jit-warning-pattern",
        action="append",
        dest="jit_warning_patterns",
        default=None,
        help="Server-log substring to count as a JIT warning; may be repeated.",
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
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        variation_prompt=args.variation_prompt,
        variation_max_tokens=args.variation_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        same_shape_followup=args.same_shape_followup,
        log_settle_seconds=args.log_settle_seconds,
        jit_warning_patterns=args.jit_warning_patterns,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
