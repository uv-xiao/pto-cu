import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_warmup_shape_probe.py"


def load_probe_module():
    assert PROBE_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_warmup_shape_probe",
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


def test_dry_run_records_identical_warmup_and_followup_shapes(tmp_path):
    probe = load_probe_module()

    result = probe.run_probe(
        artifact_dir=tmp_path / "DeepSeek-V4-Flash",
        vllm_bin=Path(".venv-vllm-probe/bin/vllm"),
        port=28126,
        server_log=tmp_path / "server.log",
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert result["server_host"] == "127.0.0.1"
    assert result["warmup_request"]["label"] == "warmup"
    assert result["followup_same_shape_request"]["label"] == "followup_same_shape"
    assert (
        result["warmup_request"]["payload"]
        == result["followup_same_shape_request"]["payload"]
    )
    assert result["warmup_request"]["payload"] == {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "prompt": "Hello",
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
    }


def test_summarize_jit_warning_windows_counts_only_selected_slices(tmp_path):
    probe = load_probe_module()
    log_path = tmp_path / "server.log"
    before = "server ready\n"
    warmup = "WARNING Triton kernel JIT compilation during inference: warmup\n"
    followup = "POST /v1/completions HTTP/1.1 200 OK\n"
    log_path.write_text(before + warmup + followup, encoding="utf-8")
    offsets = {
        "before_warmup": len(before.encode("utf-8")),
        "after_warmup": len((before + warmup).encode("utf-8")),
        "after_followup": len((before + warmup + followup).encode("utf-8")),
    }

    result = probe.summarize_jit_warning_windows(
        log_path=log_path,
        offsets=offsets,
    )

    assert result["windows"]["before_warmup"]["count"] == 0
    assert result["windows"]["warmup_request"]["count"] == 1
    assert result["windows"]["followup_same_shape_request"]["count"] == 0
    assert result["windows"]["full_before_cleanup"]["count"] == 1


def test_run_probe_cleans_up_after_failed_warmup(monkeypatch, tmp_path):
    probe = load_probe_module()
    process = FakeProcess()
    cleanup_calls = []
    log_path = tmp_path / "server.log"

    def fake_start_server(command, path):
        path.write_text("server ready\n", encoding="utf-8")
        return process, path.open("a", encoding="utf-8")

    def fake_send_contract_request(**kwargs):
        log_path.write_text(
            "server ready\n"
            "WARNING Triton kernel JIT compilation during inference: warmup\n",
            encoding="utf-8",
        )
        return {
            "status": "failed",
            "failure": {
                "category": "response_contract_usage_shape",
                "message": "usage missing",
            },
        }

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
        port=28126,
        server_log=log_path,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        request_timeout_seconds=30.0,
        terminate_timeout_seconds=0.01,
    )

    assert result["status"] == "failed"
    assert result["failure"]["category"] == "response_contract_usage_shape"
    assert result["jit_warning_summary"]["windows"]["warmup_request"]["count"] == 1
    assert result["jit_warning_summary"]["windows"]["followup_same_shape_request"][
        "count"
    ] == 0
    assert result["cleanup"]["remaining_process_group_pids"] == []
    assert cleanup_calls == [(4567, 0.01)]
