import importlib.util
import json
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_response_contract_probe.py"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_response_contract_probe", PROBE_PATH
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


def completion_payload(*, completion_tokens=2, token_ids=None):
    choice = {
        "index": 0,
        "text": "ok",
        "finish_reason": "length",
    }
    if token_ids is not None:
        choice["token_ids"] = token_ids
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "choices": [choice],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": completion_tokens,
            "total_tokens": 3 + completion_tokens,
        },
    }


def logprobs_payload(*, completion_tokens=2, prompt_tokens=3):
    payload = completion_payload(completion_tokens=completion_tokens)
    payload["usage"]["prompt_tokens"] = prompt_tokens
    payload["usage"]["total_tokens"] = prompt_tokens + completion_tokens
    payload["choices"][0]["logprobs"] = {
        "tokens": ["o", "k"][:completion_tokens],
        "token_logprobs": [-0.1, -0.2][:completion_tokens],
        "top_logprobs": [{"o": -0.1}, {"k": -0.2}][:completion_tokens],
        "text_offset": [0, 1][:completion_tokens],
    }
    payload["choices"][0]["prompt_logprobs"] = [
        None,
        {"9906": {"logprob": -0.3, "rank": 1, "decoded_token": "Hello"}},
        {"13": {"logprob": -0.4, "rank": 1, "decoded_token": "."}},
    ][:prompt_tokens]
    return payload


def test_validate_completion_contract_accepts_structurally_consistent_payload():
    probe = load_probe_module()
    request = probe.build_contract_request(max_tokens=4)

    result = probe.validate_completion_contract(
        completion_payload(completion_tokens=2, token_ids=[101, 102]),
        request=request,
    )

    assert result["status"] == "passed"
    assert result["choice_count"] == 1
    assert result["usage"]["completion_tokens"] == 2
    assert result["checks"] == {
        "choice_count": "passed",
        "choice_fields": "passed",
        "model_field": "passed",
        "token_ids": "passed",
        "usage_completion_bound": "passed",
        "usage_shape": "passed",
        "usage_total_tokens": "passed",
    }


def test_validate_completion_contract_rejects_missing_or_invalid_choice_shape():
    probe = load_probe_module()
    request = probe.build_contract_request(max_tokens=4)
    payload = completion_payload()
    payload["choices"] = []

    result = probe.validate_completion_contract(payload, request=request)

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_choice_shape"


def test_validate_completion_contract_rejects_missing_or_invalid_usage_shape():
    probe = load_probe_module()
    request = probe.build_contract_request(max_tokens=4)
    payload = completion_payload()
    payload["usage"] = {"prompt_tokens": 3, "total_tokens": 2}

    result = probe.validate_completion_contract(payload, request=request)

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_usage_shape"


def test_validate_completion_contract_rejects_token_id_count_mismatch():
    probe = load_probe_module()
    request = probe.build_contract_request(max_tokens=4)

    result = probe.validate_completion_contract(
        completion_payload(completion_tokens=2, token_ids=[101, 102, 103]),
        request=request,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_token_ids_mismatch"


def test_build_logprobs_request_adds_explicit_bounded_logprob_fields():
    probe = load_probe_module()

    request = probe.build_logprobs_contract_request(max_tokens=2)

    assert request["endpoint"] == "/v1/completions"
    assert request["payload"]["logprobs"] == 1
    assert request["payload"]["prompt_logprobs"] == 1
    assert request["payload"]["stream"] is False
    assert request["limits"]["logprobs"] == 1
    assert request["limits"]["prompt_logprobs"] == 1
    assert request["limits"]["max_tokens"] == 2


def test_validate_logprobs_contract_accepts_completion_and_prompt_shapes():
    probe = load_probe_module()
    request = probe.build_logprobs_contract_request(max_tokens=2)

    result = probe.validate_logprobs_contract(
        logprobs_payload(completion_tokens=2, prompt_tokens=3),
        request=request,
    )

    assert result["status"] == "passed"
    assert result["checks"]["base_completion_contract"] == "passed"
    assert result["checks"]["completion_logprobs_shape"] == "passed"
    assert result["checks"]["prompt_logprobs_shape"] == "passed"
    assert result["logprobs"]["completion_token_count"] == 2
    assert result["logprobs"]["prompt_logprobs_count"] == 3


def test_validate_logprobs_contract_rejects_missing_completion_logprobs():
    probe = load_probe_module()
    request = probe.build_logprobs_contract_request(max_tokens=2)
    payload = logprobs_payload(completion_tokens=2)
    payload["choices"][0]["logprobs"] = None

    result = probe.validate_logprobs_contract(payload, request=request)

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "logprobs_contract_completion_shape"


def test_validate_logprobs_contract_rejects_prompt_logprobs_beyond_usage_bound():
    probe = load_probe_module()
    request = probe.build_logprobs_contract_request(max_tokens=2)
    payload = logprobs_payload(completion_tokens=2, prompt_tokens=1)
    payload["choices"][0]["prompt_logprobs"] = [None, {"1": {"logprob": -0.1}}]

    result = probe.validate_logprobs_contract(payload, request=request)

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "logprobs_contract_prompt_bound"


def test_send_contract_request_classifies_http_failure():
    probe = load_probe_module()
    http_error = urllib.error.HTTPError(
        "http://127.0.0.1:28125/v1/completions",
        503,
        "server unavailable",
        hdrs=None,
        fp=None,
    )

    result = probe.send_contract_request(
        port=28125,
        request=probe.build_contract_request(max_tokens=4),
        timeout_seconds=30.0,
        http_post=CapturingHttpPost(http_error),
    )

    assert result["status"] == "failed"
    assert result["http_status"] == 503
    assert result["failure"]["category"] == "completion_http_error"


def test_send_contract_request_classifies_timeout():
    probe = load_probe_module()

    result = probe.send_contract_request(
        port=28125,
        request=probe.build_contract_request(max_tokens=4),
        timeout_seconds=30.0,
        http_post=CapturingHttpPost(TimeoutError("timed out")),
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "completion_timeout"


class FakeProcess:
    def __init__(self):
        self.pid = 4567
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


def test_run_probe_cleans_up_after_failed_contract_validation(monkeypatch, tmp_path):
    probe = load_probe_module()
    process = FakeProcess()
    cleanup_calls = []

    def fake_start_server(command, log_path):
        return process, tmp_path.joinpath("server.log").open("w", encoding="utf-8")

    def fake_cleanup(process, terminate_timeout_seconds):
        cleanup_calls.append((process.pid, terminate_timeout_seconds))
        return {"status": "passed", "remaining_process_group_pids": []}

    monkeypatch.setattr(probe.health_probe, "_start_server", fake_start_server)
    monkeypatch.setattr(
        probe.health_probe,
        "poll_readiness",
        lambda **kwargs: {
            "status": "passed",
            "health": {"status": "passed", "http_status": 200},
            "model_list": {"status": "passed", "http_status": 200},
            "generation_attempted": False,
        },
    )
    monkeypatch.setattr(probe.health_probe, "cleanup_process", fake_cleanup)
    monkeypatch.setattr(probe.health_probe, "_query_nvidia_smi_memory", lambda: [])
    monkeypatch.setattr(probe.health_probe, "_runtime_versions", lambda: {})
    monkeypatch.setattr(
        probe,
        "send_contract_request",
        lambda **kwargs: {
            "status": "failed",
            "failure": {
                "category": "response_contract_usage_shape",
                "message": "usage.completion_tokens is missing",
            },
        },
    )
    monkeypatch.setattr(probe.time, "monotonic", iter([0.0, 1.0]).__next__)

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28125,
        server_log=tmp_path / "server.log",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        request_timeout_seconds=30.0,
        terminate_timeout_seconds=0.01,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_usage_shape"
    assert result["cleanup"]["remaining_process_group_pids"] == []
    assert cleanup_calls == [(4567, 0.01)]
