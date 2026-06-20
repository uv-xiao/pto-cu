#!/usr/bin/env python3
"""Bounded vLLM long-prompt response-contract probe for DeepSeek V4 Flash."""

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
ADMISSION_PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_long_prompt_admission_probe.py"
)
DEFAULT_ENDPOINT = "/v1/completions"
DEFAULT_TARGET_PROMPT_TOKENS = 16_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_TOKENS = 4
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
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
    "HTTP 200 from one non-streaming /v1/completions request",
    "top-level completion response is a JSON object",
    "response model field matches served model when returned",
    "exactly one response choice object",
    "first choice exposes text and finish_reason fields",
    "generated text length is recorded without generated text contents",
    "usage prompt/completion/total token fields are internally consistent when returned",
    "usage.prompt_tokens matches measured prompt tokens when available",
    "usage.completion_tokens within request max_tokens",
    "usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens",
    "raw prompt text is not recorded",
    "raw generated text is not recorded",
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


health_probe = _load_module(
    "vllm_deepseek_v4_server_health_probe", HEALTH_PROBE_PATH
)
admission_probe = _load_module(
    "vllm_deepseek_v4_long_prompt_admission_probe", ADMISSION_PROBE_PATH
)


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
    observation: dict[str, Any] = {
        "finish_reason": choice.get("finish_reason"),
        "text_length_chars": len(choice["text"]),
    }
    if isinstance(payload.get("model"), str):
        observation["model"] = payload["model"]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        observation["usage"] = {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage.get(key), int)
        }
    return observation


def validate_long_prompt_response_contract(
    payload: Any,
    *,
    request: dict[str, Any],
    served_model_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "response_shape": _response_shape(payload),
        "checks": {},
    }
    if not isinstance(payload, dict):
        result["checks"]["payload_object"] = "failed"
        return _fail_contract(
            result,
            category="long_prompt_response_payload_shape",
            message="completion response payload is not a JSON object",
        )
    result["checks"]["payload_object"] = "passed"

    model = payload.get("model")
    if model is None:
        result["checks"]["model_field"] = "not_returned"
    elif model != served_model_name:
        result["checks"]["model_field"] = "failed"
        return _fail_contract(
            result,
            category="long_prompt_response_model_mismatch",
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
            category="long_prompt_response_choice_shape",
            message="completion response must contain exactly one object choice",
        )
    result["checks"]["choice_count"] = "passed"
    result["choice_count"] = 1

    choice = choices[0]
    if "finish_reason" not in choice or not isinstance(choice.get("text"), str):
        result["checks"]["choice_fields"] = "failed"
        return _fail_contract(
            result,
            category="long_prompt_response_choice_fields",
            message="completion choice must include finish_reason and string text",
        )
    result["checks"]["choice_fields"] = "passed"
    result["checks"]["generated_text_recorded"] = "length_only"
    result["text_length_chars"] = len(choice["text"])

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="long_prompt_response_usage_shape",
            message="completion response must include a usage object",
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    usage_values = (prompt_tokens, completion_tokens, total_tokens)
    if not all(isinstance(value, int) for value in usage_values):
        result["checks"]["usage_shape"] = "failed"
        return _fail_contract(
            result,
            category="long_prompt_response_usage_shape",
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
            category="long_prompt_response_prompt_token_mismatch",
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
            category="long_prompt_response_completion_bound",
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
            category="long_prompt_response_usage_total_tokens",
            message=(
                "usage.total_tokens must be greater than or equal to "
                "usage.prompt_tokens + usage.completion_tokens"
            ),
        )
    result["checks"]["usage_total_tokens"] = "passed"
    return result


def build_long_prompt_contract_request(
    *,
    served_model_name: str = health_probe.DEFAULT_SERVED_MODEL_NAME,
    artifact_dir: Path = health_probe.DEFAULT_ARTIFACT_DIR,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if max_tokens != DEFAULT_MAX_TOKENS:
        raise ValueError(
            f"long-prompt response-contract requests require max_tokens={DEFAULT_MAX_TOKENS}"
        )
    prompt_result = admission_probe.build_synthetic_prompt(
        artifact_dir=artifact_dir,
        target_prompt_tokens=target_prompt_tokens,
    )
    payload = {
        "model": served_model_name,
        "prompt": prompt_result["prompt"],
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
            **prompt_result["accounting"],
            "max_tokens": max_tokens,
            "stream": False,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "echo": False,
            "logprobs": False,
        },
    }


def build_planned_long_prompt_contract_request(
    *,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if target_prompt_tokens < 1:
        raise ValueError("target prompt-token budget must be positive")
    if max_tokens != DEFAULT_MAX_TOKENS:
        raise ValueError(
            f"long-prompt response-contract requests require max_tokens={DEFAULT_MAX_TOKENS}"
        )
    return {
        "endpoint": DEFAULT_ENDPOINT,
        "limits": {
            "target_prompt_tokens": target_prompt_tokens,
            "actual_prompt_tokens": None,
            "tokenizer_accounting": "planned",
            "prompt_chars": None,
            "prompt_unit_chars": len(admission_probe.DEFAULT_PROMPT_UNIT),
            "max_tokens": max_tokens,
            "stream": False,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "echo": False,
            "logprobs": False,
        },
    }


def _review_safe_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": request["endpoint"],
        "limits": request["limits"],
        "prompt_text_recorded": False,
        "payload_recorded": False,
    }


def _find_served_model_entry(model_list: dict[str, Any], served_model_name: str) -> Any:
    raw = model_list.get("raw")
    if not isinstance(raw, dict):
        return None
    data = raw.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and item.get("id") == served_model_name:
            return item
    return None


def validate_readiness_contract(
    readiness: dict[str, Any],
    *,
    served_model_name: str,
    max_model_len: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "checks": {},
    }
    health = readiness.get("health")
    if not isinstance(health, dict) or health.get("http_status") != 200:
        result["checks"]["health_http_200"] = "failed"
        return _fail_contract(
            result,
            category="readiness_health_http_status",
            message="/health did not return HTTP 200",
        )
    result["checks"]["health_http_200"] = "passed"

    model_list = readiness.get("model_list")
    if not isinstance(model_list, dict) or model_list.get("http_status") != 200:
        result["checks"]["models_http_200"] = "failed"
        return _fail_contract(
            result,
            category="readiness_models_http_status",
            message="/v1/models did not return HTTP 200",
        )
    result["checks"]["models_http_200"] = "passed"

    entry = _find_served_model_entry(model_list, served_model_name)
    if not isinstance(entry, dict):
        result["checks"]["served_model_listed"] = "failed"
        return _fail_contract(
            result,
            category="readiness_served_model_missing",
            message=f"/v1/models did not include {served_model_name}",
        )
    result["checks"]["served_model_listed"] = "passed"
    result["served_model"] = {
        "id": entry.get("id"),
        "max_model_len": entry.get("max_model_len"),
    }

    if entry.get("max_model_len") != max_model_len:
        result["checks"]["max_model_len"] = "failed"
        return _fail_contract(
            result,
            category="readiness_max_model_len_mismatch",
            message=(
                f"/v1/models max_model_len={entry.get('max_model_len')} does not "
                f"match expected {max_model_len}"
            ),
        )
    result["checks"]["max_model_len"] = "passed"
    return result


def send_contract_request(
    *,
    port: int,
    request: dict[str, Any],
    served_model_name: str,
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

    contract = validate_long_prompt_response_contract(
        payload,
        request=request,
        served_model_name=served_model_name,
    )
    result["contract"] = contract
    result["response_shape"] = contract["response_shape"]
    result["status"] = contract["status"]
    if contract["status"] != "passed":
        result["failure"] = contract["failure"]
    else:
        result["observation"] = _completion_observation(payload)
    return result


def _server_log_snippets(log_path: Path, *, max_lines: int = 80) -> list[str]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


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
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        "failure_category": failure.get("category", "unknown"),
        "failure_message": failure.get("message", ""),
        "server_log_tail": _server_log_snippets(log_path),
        "cleanup": result.get("cleanup", {}),
        "next_diagnostic_gate": (
            "rerun the same long-prompt response-contract probe after checking "
            "server readiness, model-list metadata, and completion usage accounting"
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
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    terminate_timeout_seconds: float = health_probe.DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
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
            contract_request = build_planned_long_prompt_contract_request(
                target_prompt_tokens=target_prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        else:
            contract_request = build_long_prompt_contract_request(
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
        / "vllm-long-prompt-response-contract-probe"
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
        "request": _review_safe_request(contract_request),
        "contract_checks": CONTRACT_CHECKS,
        "request_timeout_seconds": request_timeout_seconds,
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
        readiness_contract = validate_readiness_contract(
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
        completion = send_contract_request(
            port=selected_port,
            request=contract_request,
            served_model_name=served_model_name,
            timeout_seconds=request_timeout_seconds,
        )
        result["generation_attempted"] = True
        result["prompt_sent"] = True
        result["completion"] = completion
        result["status"] = completion["status"]
        if completion["status"] != "passed":
            result["failure"] = completion.get(
                "failure",
                _failure_payload(
                    "long_prompt_response_contract_failed",
                    "long-prompt response contract failed",
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
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
