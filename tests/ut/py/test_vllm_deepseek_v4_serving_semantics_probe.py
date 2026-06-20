import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_serving_semantics_probe.py"
)


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_serving_semantics_probe",
        PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self):
        self.pid = 4567
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


def passed_completion(*, digest="same", text_length=12, finish_reason="length"):
    return {
        "status": "passed",
        "contract": {
            "status": "passed",
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 8,
                "total_tokens": 17,
            },
        },
        "observation": {
            "text_sha256": digest,
            "text_length_chars": text_length,
            "finish_reason": finish_reason,
            "stop_reason": None,
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 8,
                "total_tokens": 17,
            },
        },
    }


def test_dry_run_records_identical_deterministic_requests(tmp_path):
    probe = load_probe_module()

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28128,
        server_log=tmp_path / "server.log",
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["first_request"]["label"] == "first_deterministic"
    assert result["repeat_request"]["label"] == "repeat_deterministic"
    assert result["first_request"]["payload"] == result["repeat_request"]["payload"]
    assert result["first_request"]["payload"]["temperature"] == 0.0
    assert result["first_request"]["payload"]["top_p"] == 1.0
    assert result["first_request"]["payload"]["seed"] == 0
    assert result["first_request"]["payload"]["stream"] is False
    assert result["first_request"]["payload"]["max_tokens"] == 8


def test_compare_deterministic_observations_accepts_matching_shape():
    probe = load_probe_module()
    request = probe.build_semantics_request()

    result = probe.compare_deterministic_observations(
        passed_completion(),
        passed_completion(),
        request=request,
    )

    assert result["status"] == "passed"
    assert result["checks"] == {
        "finish_reason": "passed",
        "text_digest": "passed",
        "text_length": "passed",
        "usage": "passed",
    }
    assert result["observed"]["text_length_chars"] == 12
    assert result["observed"]["completion_tokens"] == 8


def test_compare_deterministic_observations_rejects_digest_mismatch():
    probe = load_probe_module()
    request = probe.build_semantics_request()

    result = probe.compare_deterministic_observations(
        passed_completion(digest="first"),
        passed_completion(digest="second"),
        request=request,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "serving_semantics_text_digest"


def test_run_probe_cleans_up_after_determinism_failure(monkeypatch, tmp_path):
    probe = load_probe_module()
    process = FakeProcess()
    cleanup_calls = []
    log_path = tmp_path / "server.log"
    responses = [
        passed_completion(digest="first"),
        passed_completion(digest="second"),
    ]

    def fake_start_server(command, path):
        path.write_text("server ready\n", encoding="utf-8")
        return process, path.open("a", encoding="utf-8")

    def fake_send_contract_request(**kwargs):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("POST /v1/completions HTTP/1.1 200 OK\n")
        return responses.pop(0)

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
        probe.response_contract_probe,
        "send_contract_request",
        fake_send_contract_request,
    )
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(probe.time, "monotonic", iter([0.0, 1.0]).__next__)

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28128,
        server_log=log_path,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        request_timeout_seconds=30.0,
        terminate_timeout_seconds=0.01,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "serving_semantics_text_digest"
    assert result["first_completion"]["status"] == "passed"
    assert result["repeat_completion"]["status"] == "passed"
    assert result["cleanup"]["remaining_process_group_pids"] == []
    assert cleanup_calls == [(4567, 0.01)]
