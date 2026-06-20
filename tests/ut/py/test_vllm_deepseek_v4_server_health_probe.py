import importlib.util
import json
import subprocess
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_server_health_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_server_health_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_server_command_binds_localhost_and_uses_inspected_vllm_flags(tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"

    command = probe.build_server_command(
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        artifact_dir=artifact_dir,
        port=28123,
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        max_model_len=4096,
        tensor_parallel_size=2,
        dtype="bfloat16",
        quantization="deepseek_v4_fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.78,
        distributed_executor_backend="mp",
        enforce_eager=True,
        trust_remote_code=False,
    )

    assert command == [
        ".venv-vllm-probe/bin/vllm",
        "serve",
        str(artifact_dir),
        "--host",
        "127.0.0.1",
        "--port",
        "28123",
        "--served-model-name",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--tokenizer",
        str(artifact_dir),
        "--tokenizer-mode",
        "deepseek_v4",
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--quantization",
        "deepseek_v4_fp8",
        "--kv-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.78",
        "--distributed-executor-backend",
        "mp",
        "--enforce-eager",
    ]


class FakeResponse:
    def __init__(self, status, body=b"{}"):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


class SequencedHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_poll_endpoints_waits_for_health_then_reads_model_list(monkeypatch):
    probe = load_probe_module()
    http_client = SequencedHttpClient(
        [
            OSError("not ready"),
            FakeResponse(200, b"ok"),
            FakeResponse(
                200,
                json.dumps({"data": [{"id": "deepseek-ai/DeepSeek-V4-Flash"}]}).encode(),
            ),
        ]
    )
    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

    result = probe.poll_readiness(
        port=28123,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        http_get=http_client,
    )

    assert result["status"] == "passed"
    assert result["health"]["status"] == "passed"
    assert result["model_list"]["status"] == "passed"
    assert result["model_list"]["model_ids"] == ["deepseek-ai/DeepSeek-V4-Flash"]
    assert http_client.urls == [
        "http://127.0.0.1:28123/health",
        "http://127.0.0.1:28123/health",
        "http://127.0.0.1:28123/v1/models",
    ]


def test_poll_endpoints_classifies_timeout(monkeypatch):
    probe = load_probe_module()
    http_client = SequencedHttpClient([OSError("refused"), OSError("refused")])
    times = iter([0.0, 1.1])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

    result = probe.poll_readiness(
        port=28123,
        timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        http_get=http_client,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "timeout"
    assert result["health"]["last_error"] == "OSError: refused"


class FakeProcess:
    def __init__(self, returncode=None):
        self.pid = 4567
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


def test_cleanup_terminates_server_and_reports_no_remaining_processes(monkeypatch):
    probe = load_probe_module()
    process = FakeProcess()
    monkeypatch.setattr(probe, "_remaining_process_group_pids", lambda pid: [])

    result = probe.cleanup_process(process, terminate_timeout_seconds=0.01)

    assert process.terminated is True
    assert process.killed is False
    assert result["status"] == "passed"
    assert result["terminated"] is True
    assert result["killed"] is False
    assert result["remaining_process_group_pids"] == []


def test_cleanup_waits_for_process_group_to_drain(monkeypatch):
    probe = load_probe_module()
    process = FakeProcess()
    remaining = iter([[111, 222], []])
    sleeps = []
    monkeypatch.setattr(probe, "_remaining_process_group_pids", lambda pgid: next(remaining))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(probe.time, "monotonic", iter([0.0, 0.1]).__next__)

    result = probe.cleanup_process(process, terminate_timeout_seconds=1.0)

    assert result["status"] == "passed"
    assert result["remaining_process_group_pids"] == []
    assert sleeps == [0.25]


def test_model_list_404_is_recorded_as_unavailable_not_generation_requirement(monkeypatch):
    probe = load_probe_module()
    http_client = SequencedHttpClient(
        [
            FakeResponse(200, b"ok"),
            urllib.error.HTTPError(
                "http://127.0.0.1:28123/v1/models",
                404,
                "not found",
                hdrs=None,
                fp=None,
            ),
        ]
    )
    times = iter([0.0, 0.1])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))

    result = probe.poll_readiness(
        port=28123,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        http_get=http_client,
    )

    assert result["status"] == "passed"
    assert result["model_list"]["status"] == "unavailable"
    assert result["model_list"]["http_status"] == 404
    assert result["generation_attempted"] is False
