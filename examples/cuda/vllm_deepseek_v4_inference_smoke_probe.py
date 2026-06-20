#!/usr/bin/env python3
"""Bounded one-token vLLM inference smoke probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
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
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 1
DEFAULT_TEMPERATURE = 0.0
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


def build_inference_request(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    if endpoint not in {"/v1/completions", "/v1/chat/completions"}:
        raise ValueError(f"unsupported inference endpoint: {endpoint}")
    if max_tokens < 0 or max_tokens > 1:
        raise ValueError("inference smoke requests at most one output token")
    payload: dict[str, Any] = {
        "model": served_model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if endpoint == "/v1/chat/completions":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    return {
        "endpoint": endpoint,
        "payload": payload,
        "limits": {
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt),
            "stream": False,
        },
    }


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


def send_inference_request(
    *,
    port: int,
    request: dict[str, Any],
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    http_post: HttpPost = _http_post,
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
                    "inference_http_status",
                    f"inference endpoint returned HTTP {status}",
                )
                return result
            payload = _read_json_response(response)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["failure"] = _failure_payload(
            "inference_http_error",
            _exception_summary(exc),
        )
        return result
    except TimeoutError as exc:
        result["failure"] = _failure_payload(
            "inference_timeout",
            _exception_summary(exc),
        )
        return result
    except Exception as exc:
        result["failure"] = _failure_payload(
            "inference_request_error",
            _exception_summary(exc),
        )
        return result
    result["status"] = "passed"
    result["response_shape"] = _response_shape(payload)
    return result


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
    endpoint: str = DEFAULT_ENDPOINT,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
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
    try:
        inference_request = build_inference_request(
            endpoint=endpoint,
            served_model_name=served_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
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
        / "vllm-inference-smoke-probe"
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
        "endpoints": ["/health", "/v1/models", endpoint],
        "inference_request": inference_request,
        "request_timeout_seconds": request_timeout_seconds,
        "runtime_versions": health_probe._runtime_versions(),
        "gpu_memory_before": health_probe._query_nvidia_smi_memory(),
        "generation_attempted": False,
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
        inference = send_inference_request(
            port=selected_port,
            request=inference_request,
            timeout_seconds=request_timeout_seconds,
        )
        result["generation_attempted"] = True
        result["inference"] = inference
        result["status"] = inference["status"]
        if inference["status"] != "passed":
            result["failure"] = inference.get(
                "failure",
                _failure_payload("inference_failed", "inference request failed"),
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
    parser.add_argument(
        "--endpoint",
        choices=("/v1/completions", "/v1/chat/completions"),
        default=DEFAULT_ENDPOINT,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
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
        endpoint=args.endpoint,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
