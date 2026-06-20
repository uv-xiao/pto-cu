import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = (
    ROOT / "examples" / "cuda" / "vllm_deepseek_v4_request_shape_variation_probe.py"
)


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_request_shape_variation_probe",
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


def passed_completion():
    return {
        "status": "passed",
        "contract": {
            "status": "passed",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 4,
                "total_tokens": 5,
            },
        },
    }


def test_dry_run_records_warmup_baseline_and_distinct_variation_shape(tmp_path):
    probe = load_probe_module()

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28127,
        server_log=tmp_path / "server.log",
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["warmup_request"]["label"] == "warmup"
    assert result["followup_same_shape_request"]["label"] == "followup_same_shape"
    assert result["variation_request"]["label"] == "variation"
    assert (
        result["warmup_request"]["payload"]
        == result["followup_same_shape_request"]["payload"]
    )
    assert result["warmup_request"]["payload"]["prompt"] == "Hello"
    assert result["warmup_request"]["payload"]["max_tokens"] == 4
    assert result["variation_request"]["payload"]["prompt"] != "Hello"
    assert result["variation_request"]["payload"]["max_tokens"] == 8
    assert result["variation_request"]["limits"]["prompt_chars"] > 5


def test_summarize_jit_warning_windows_counts_variation_slice(tmp_path):
    probe = load_probe_module()
    log_path = tmp_path / "server.log"
    before = "server ready\n"
    warmup = "WARNING Triton kernel JIT compilation during inference: warmup\n"
    followup = "POST /v1/completions HTTP/1.1 200 OK\n"
    variation = "WARNING Triton kernel JIT compilation during inference: variation\n"
    log_path.write_text(before + warmup + followup + variation, encoding="utf-8")
    offsets = {
        "before_warmup": len(before.encode("utf-8")),
        "after_warmup": len((before + warmup).encode("utf-8")),
        "after_followup": len((before + warmup + followup).encode("utf-8")),
        "after_variation": len(
            (before + warmup + followup + variation).encode("utf-8")
        ),
    }

    result = probe.summarize_jit_warning_windows(
        log_path=log_path,
        offsets=offsets,
    )

    assert result["windows"]["before_warmup"]["count"] == 0
    assert result["windows"]["warmup_request"]["count"] == 1
    assert result["windows"]["followup_same_shape_request"]["count"] == 0
    assert result["windows"]["variation_request"]["count"] == 1
    assert result["windows"]["full_before_cleanup"]["count"] == 2


def test_run_probe_cleans_up_after_failed_variation(monkeypatch, tmp_path):
    probe = load_probe_module()
    process = FakeProcess()
    cleanup_calls = []
    log_path = tmp_path / "server.log"
    responses = [
        passed_completion(),
        passed_completion(),
        {
            "status": "failed",
            "failure": {
                "category": "response_contract_usage_shape",
                "message": "usage missing",
            },
        },
    ]

    def fake_start_server(command, path):
        path.write_text("server ready\n", encoding="utf-8")
        return process, path.open("a", encoding="utf-8")

    def fake_send_contract_request(**kwargs):
        request = kwargs["request"]
        label = (
            "variation"
            if request["payload"]["prompt"] == probe.DEFAULT_VARIATION_PROMPT
            else "same_shape"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"WARNING Triton kernel JIT compilation during inference: {label}\n"
            )
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
        port=28127,
        server_log=log_path,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        request_timeout_seconds=30.0,
        terminate_timeout_seconds=0.01,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_usage_shape"
    assert result["warmup_completion"]["status"] == "passed"
    assert result["followup_same_shape_completion"]["status"] == "passed"
    assert result["variation_completion"]["status"] == "failed"
    assert result["jit_warning_summary"]["windows"]["variation_request"]["count"] == 1
    assert result["cleanup"]["remaining_process_group_pids"] == []
    assert cleanup_calls == [(4567, 0.01)]


def test_run_probe_rejects_unbounded_variation_token_request(tmp_path):
    probe = load_probe_module()

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28127,
        server_log=tmp_path / "server.log",
        variation_max_tokens=17,
        dry_run=True,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "invalid_variation_request"
