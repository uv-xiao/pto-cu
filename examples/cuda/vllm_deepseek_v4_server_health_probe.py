#!/usr/bin/env python3
"""Bounded vLLM OpenAI server health probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = (
    ROOT / "tmp" / "model-artifacts" / "deepseek-ai" / "DeepSeek-V4-Flash"
)
DEFAULT_VLLM_BIN = ROOT / ".venv-vllm-probe" / "bin" / "vllm"
DEFAULT_SERVED_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_DTYPE = "bfloat16"
DEFAULT_QUANTIZATION = "deepseek_v4_fp8"
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.78
DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND = "mp"
DEFAULT_TIMEOUT_SECONDS = 2700.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_TERMINATE_TIMEOUT_SECONDS = 30.0
LOCAL_HOST = "127.0.0.1"
NON_CLAIMS = [
    "not generated-text correctness evidence",
    "not tokenizer semantic correctness evidence",
    "not prompt correctness evidence",
    "not 256K context evidence",
    "not throughput or latency evidence",
    "not production-readiness evidence",
    "not simpler-nv or vLLM kernel integration evidence",
]

HttpGet = Callable[[str, float], Any]


def _display_path(path: Path) -> str:
    absolute_path = path if path.is_absolute() else ROOT / path
    try:
        return str(absolute_path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _command_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def choose_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCAL_HOST, 0))
        return int(sock.getsockname()[1])


def build_server_command(
    *,
    vllm_bin: Path,
    artifact_dir: Path,
    port: int,
    served_model_name: str,
    max_model_len: int,
    tensor_parallel_size: int,
    dtype: str,
    quantization: str | None,
    kv_cache_dtype: str,
    gpu_memory_utilization: float,
    distributed_executor_backend: str,
    enforce_eager: bool,
    trust_remote_code: bool,
) -> list[str]:
    command = [
        _command_path(vllm_bin),
        "serve",
        str(artifact_dir),
        "--host",
        LOCAL_HOST,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        str(artifact_dir),
        "--tokenizer-mode",
        "deepseek_v4",
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--dtype",
        dtype,
    ]
    if quantization:
        command.extend(["--quantization", quantization])
    command.extend(
        [
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--distributed-executor-backend",
            distributed_executor_backend,
        ]
    )
    command.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _query_nvidia_smi_memory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return [{"error": completed.stderr.strip()}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": parts[2],
                "memory_free_mib": parts[3],
                "memory_total_mib": parts[4],
            }
        )
    return rows


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for module_name in ("vllm", "torch"):
        try:
            module = __import__(module_name)
        except Exception:
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
        if module_name == "torch":
            versions["torch_cuda"] = str(getattr(module.version, "cuda", None))
    return versions


def _http_get(url: str, timeout: float):
    request = urllib.request.Request(url, method="GET")
    return urllib.request.urlopen(request, timeout=timeout)


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _failure_payload(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _read_json_response(response: Any) -> Any:
    data = response.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def _read_model_list(port: int, http_get: HttpGet, request_timeout: float) -> dict[str, Any]:
    url = f"http://{LOCAL_HOST}:{port}/v1/models"
    try:
        with http_get(url, request_timeout) as response:
            status = int(getattr(response, "status", 0))
            if status != 200:
                return {
                    "status": "failed",
                    "endpoint": "/v1/models",
                    "http_status": status,
                }
            payload = _read_json_response(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "status": "unavailable",
                "endpoint": "/v1/models",
                "http_status": exc.code,
            }
        return {
            "status": "failed",
            "endpoint": "/v1/models",
            "http_status": exc.code,
            "error": _exception_summary(exc),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "endpoint": "/v1/models",
            "error": _exception_summary(exc),
        }
    return {
        "status": "passed",
        "endpoint": "/v1/models",
        "http_status": 200,
        "model_ids": [item.get("id") for item in payload.get("data", [])],
        "raw": payload,
    }


def poll_readiness(
    *,
    port: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
    http_get: HttpGet = _http_get,
    request_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://{LOCAL_HOST}:{port}/health"
    health: dict[str, Any] = {
        "status": "pending",
        "endpoint": "/health",
        "url": health_url,
        "attempts": 0,
    }
    while True:
        health["attempts"] += 1
        try:
            with http_get(health_url, request_timeout_seconds) as response:
                status = int(getattr(response, "status", 0))
                health["http_status"] = status
                if status == 200:
                    health["status"] = "passed"
                    break
                health["last_error"] = f"HTTP status {status}"
        except Exception as exc:
            health["last_error"] = _exception_summary(exc)

        if time.monotonic() >= deadline:
            return {
                "status": "failed",
                "health": health,
                "model_list": {"status": "not_attempted", "endpoint": "/v1/models"},
                "generation_attempted": False,
                "failure": _failure_payload(
                    "timeout",
                    f"server did not pass /health within {timeout_seconds} seconds",
                ),
            }
        time.sleep(poll_interval_seconds)

    model_list = _read_model_list(port, http_get, request_timeout_seconds)
    status = "passed" if model_list["status"] in {"passed", "unavailable"} else "failed"
    result: dict[str, Any] = {
        "status": status,
        "health": health,
        "model_list": model_list,
        "generation_attempted": False,
    }
    if status == "failed":
        result["failure"] = _failure_payload(
            "model_list_failed",
            "server passed /health but /v1/models did not return a usable response",
        )
    return result


def _remaining_process_group_pids(pgid: int) -> list[int]:
    completed = subprocess.run(
        ["ps", "-o", "pid=", "-g", str(pgid)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return []
    current_pid = os.getpid()
    pids = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            remaining = int(line)
        except ValueError:
            continue
        if remaining != current_pid:
            pids.append(remaining)
    return sorted(pids)


def _wait_for_process_group_empty(
    pgid: int,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.25,
) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = _remaining_process_group_pids(pgid)
        if not remaining:
            return []
        if time.monotonic() >= deadline:
            return remaining
        time.sleep(poll_interval_seconds)


def cleanup_process(
    process: subprocess.Popen,
    *,
    terminate_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        pgid = process.pid
    result: dict[str, Any] = {
        "pid": process.pid,
        "process_group_id": pgid,
        "terminated": False,
        "killed": False,
        "returncode_before_cleanup": process.poll(),
    }
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.terminate()
        except OSError:
            process.terminate()
        result["terminated"] = True
        try:
            process.wait(timeout=terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            except OSError:
                process.kill()
            result["killed"] = True
            process.wait(timeout=terminate_timeout_seconds)
    else:
        process.wait(timeout=terminate_timeout_seconds)
    remaining = _wait_for_process_group_empty(
        pgid,
        timeout_seconds=terminate_timeout_seconds,
    )
    if remaining and not result["killed"]:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        result["killed"] = True
        remaining = _wait_for_process_group_empty(
            pgid,
            timeout_seconds=terminate_timeout_seconds,
        )
    result["returncode_after_cleanup"] = process.returncode
    result["remaining_process_group_pids"] = remaining
    result["status"] = "passed" if not remaining else "failed"
    return result


def _start_server(command: list[str], log_path: Path) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_file


def run_probe(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    vllm_bin: Path = DEFAULT_VLLM_BIN,
    port: int | None = None,
    server_log: Path | None = None,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    dtype: str = DEFAULT_DTYPE,
    quantization: str | None = DEFAULT_QUANTIZATION,
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    distributed_executor_backend: str = DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    enforce_eager: bool = True,
    trust_remote_code: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    terminate_timeout_seconds: float = DEFAULT_TERMINATE_TIMEOUT_SECONDS,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_port = port if port is not None else choose_local_port()
    if selected_port <= 0 or selected_port > 65535:
        return {
            "status": "failed",
            "failure": _failure_payload("invalid_port", f"invalid port: {selected_port}"),
            "generation_attempted": False,
            "non_claims": NON_CLAIMS,
        }
    log_path = server_log or (
        ROOT
        / "tmp"
        / "vllm-server-health-probe"
        / f"server-{selected_port}.log"
    )
    command = build_server_command(
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
        "server_host": LOCAL_HOST,
        "server_port": selected_port,
        "server_log": _display_path(log_path),
        "endpoints": ["/health", "/v1/models"],
        "runtime_versions": _runtime_versions(),
        "gpu_memory_before": _query_nvidia_smi_memory(),
        "generation_attempted": False,
        "non_claims": NON_CLAIMS,
    }
    if dry_run:
        return result

    started = time.monotonic()
    process = None
    log_file = None
    try:
        process, log_file = _start_server(command, log_path)
        result["server_pid"] = process.pid
        readiness = poll_readiness(
            port=selected_port,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        result["readiness"] = readiness
        result["status"] = readiness["status"]
        if process.poll() is not None and result["status"] != "passed":
            result["failure"] = _failure_payload(
                "server_exited_before_ready",
                f"server process exited with {process.returncode}",
            )
        elif "failure" in readiness:
            result["failure"] = readiness["failure"]
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = _failure_payload("probe_error", _exception_summary(exc))
    finally:
        if process is not None:
            result["cleanup"] = cleanup_process(
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
        result["gpu_memory_after"] = _query_nvidia_smi_memory()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--vllm-bin", type=Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--kv-cache-dtype", default=DEFAULT_KV_CACHE_DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--distributed-executor-backend",
        default=DEFAULT_DISTRIBUTED_EXECUTOR_BACKEND,
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--terminate-timeout-seconds",
        type=float,
        default=DEFAULT_TERMINATE_TIMEOUT_SECONDS,
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
        terminate_timeout_seconds=args.terminate_timeout_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
