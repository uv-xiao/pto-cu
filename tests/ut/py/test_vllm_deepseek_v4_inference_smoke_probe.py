import importlib.util
import json
import subprocess
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_inference_smoke_probe.py"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_inference_smoke_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


class CapturingHttpPost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_build_inference_request_uses_one_token_completion_defaults():
    probe = load_probe_module()

    request = probe.build_inference_request(
        endpoint="/v1/completions",
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
    )

    assert request == {
        "endpoint": "/v1/completions",
        "payload": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
        "limits": {
            "max_tokens": 1,
            "prompt_chars": 5,
            "stream": False,
        },
    }


def test_build_inference_request_rejects_more_than_one_output_token():
    probe = load_probe_module()

    try:
        probe.build_inference_request(
            endpoint="/v1/completions",
            served_model_name="deepseek-ai/DeepSeek-V4-Flash",
            prompt="Hello",
            max_tokens=2,
            temperature=0.0,
        )
    except ValueError as exc:
        assert "at most one output token" in str(exc)
    else:
        raise AssertionError("expected max_tokens > 1 to fail")


def test_build_inference_request_rejects_unknown_endpoint():
    probe = load_probe_module()

    try:
        probe.build_inference_request(
            endpoint="/v1/embeddings",
            served_model_name="deepseek-ai/DeepSeek-V4-Flash",
            prompt="Hello",
            max_tokens=1,
            temperature=0.0,
        )
    except ValueError as exc:
        assert "unsupported inference endpoint" in str(exc)
    else:
        raise AssertionError("expected unsupported endpoint to fail")


def test_send_inference_request_posts_to_local_completion_endpoint():
    probe = load_probe_module()
    http_post = CapturingHttpPost(
        FakeResponse(
            200,
            json.dumps(
                {
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "choices": [{"text": "x", "finish_reason": "length"}],
                    "usage": {"completion_tokens": 1},
                }
            ).encode(),
        )
    )
    request = probe.build_inference_request(
        endpoint="/v1/completions",
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
    )

    result = probe.send_inference_request(
        port=28123,
        request=request,
        timeout_seconds=30.0,
        http_post=http_post,
    )

    assert result["status"] == "passed"
    assert result["endpoint"] == "/v1/completions"
    assert result["http_status"] == 200
    assert result["response_shape"] == {
        "top_level_keys": ["choices", "id", "object", "usage"],
        "choice_count": 1,
        "first_choice_keys": ["finish_reason", "text"],
        "usage_keys": ["completion_tokens"],
    }
    assert http_post.calls == [
        (
            "http://127.0.0.1:28123/v1/completions",
            request["payload"],
            30.0,
        )
    ]


def test_send_inference_request_classifies_http_failure():
    probe = load_probe_module()
    http_error = urllib.error.HTTPError(
        "http://127.0.0.1:28123/v1/completions",
        500,
        "server error",
        hdrs=None,
        fp=None,
    )
    request = probe.build_inference_request(
        endpoint="/v1/completions",
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
    )

    result = probe.send_inference_request(
        port=28123,
        request=request,
        timeout_seconds=30.0,
        http_post=CapturingHttpPost(http_error),
    )

    assert result["status"] == "failed"
    assert result["http_status"] == 500
    assert result["failure"]["category"] == "inference_http_error"


def test_send_inference_request_classifies_timeout():
    probe = load_probe_module()
    request = probe.build_inference_request(
        endpoint="/v1/completions",
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt="Hello",
        max_tokens=1,
        temperature=0.0,
    )

    result = probe.send_inference_request(
        port=28123,
        request=request,
        timeout_seconds=30.0,
        http_post=CapturingHttpPost(TimeoutError("timed out")),
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "inference_timeout"


class FakeProcess:
    def __init__(self):
        self.pid = 4567
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


def test_run_probe_attempts_inference_after_readiness_and_cleans_up(monkeypatch, tmp_path):
    probe = load_probe_module()
    process = FakeProcess()
    started_commands = []

    def fake_start_server(command, log_path):
        started_commands.append((command, log_path))
        return process, tmp_path.joinpath("server.log").open("w", encoding="utf-8")

    monkeypatch.setattr(probe.health_probe, "_start_server", fake_start_server)
    monkeypatch.setattr(
        probe.health_probe,
        "poll_readiness",
        lambda **kwargs: {
            "status": "passed",
            "health": {"status": "passed"},
            "model_list": {"status": "passed"},
            "generation_attempted": False,
        },
    )
    monkeypatch.setattr(
        probe.health_probe,
        "cleanup_process",
        lambda process, terminate_timeout_seconds: {
            "status": "passed",
            "remaining_process_group_pids": [],
        },
    )
    monkeypatch.setattr(probe.health_probe, "_query_nvidia_smi_memory", lambda: [])
    monkeypatch.setattr(probe.health_probe, "_runtime_versions", lambda: {})
    monkeypatch.setattr(probe, "send_inference_request", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(probe.time, "monotonic", iter([0.0, 1.0]).__next__)

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28123,
        server_log=tmp_path / "server.log",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        request_timeout_seconds=30.0,
        terminate_timeout_seconds=0.01,
    )

    assert result["status"] == "passed"
    assert result["readiness"]["status"] == "passed"
    assert result["inference"]["status"] == "passed"
    assert result["generation_attempted"] is True
    assert result["cleanup"]["remaining_process_group_pids"] == []
    assert started_commands[0][0][0:4] == [
        ".venv-vllm-probe/bin/vllm",
        "serve",
        str(tmp_path / "DeepSeek-V4-Flash"),
        "--host",
    ]
