"""Tests for the synthetic pypto-serving simpler-nv shim."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import builtins
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_serving_shim_example():
    example = ROOT / "examples" / "cuda" / "pypto_serving_nv_shim.py"
    assert example.is_file(), f"missing {example.relative_to(ROOT)}"
    spec = importlib.util.spec_from_file_location("pypto_serving_nv_shim_example", example)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_serving_request_uses_simpler_nv_executor_boundary():
    module = _load_serving_shim_example()
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    result = module.run_synthetic_serving_request(
        prompt="hello",
        max_new_tokens=2,
        device_id=3,
        kernel_launcher=record_launch,
    )

    assert result["status"] == "passed"
    assert result["backend"] == "simpler-nv"
    assert result["model_id"] == "synthetic-simpler-nv"
    assert result["text"] == "NV"
    assert result["token_ids"] == [1, 2]
    assert result["finish_reason"] == "length"
    assert result["launch_count"] == 2
    assert [launch.phase for launch in launches] == ["prefill", "decode"]
    assert {launch.platform for launch in launches} == {"cuda"}
    assert {launch.runtime for launch in launches} == {"host_schedule"}
    assert {launch.device_id for launch in launches} == {3}


def test_simpler_nv_executor_registers_runner_and_generates_logits():
    module = _load_serving_shim_example()
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase}

    model = module.create_synthetic_runtime_model()
    executor = module.SimplerNvExecutor(
        platform="cuda",
        runtime="host_schedule",
        device_id=0,
        kernel_launcher=record_launch,
    )
    executor.register_model(model.model_id, model)

    prefill = executor.run_prefill(
        model,
        module.PrefillBatch(request_ids=["req-0"], token_ids=[0]),
    )
    decode = executor.run_decode(
        model,
        module.DecodeBatch(request_ids=["req-0"], token_ids=[1], hidden_states=prefill.last_hidden),
    )

    assert prefill.logits == [[0.0, 10.0, 0.0]]
    assert decode.logits == [[0.0, 0.0, 10.0]]
    assert prefill.last_hidden == [[1.0, 0.0, 0.0, 0.0]]
    assert decode.hidden_states == [[0.0, 1.0, 0.0, 0.0]]
    assert [launch.phase for launch in launches] == ["prefill", "decode"]
    assert executor.runner_for(model.model_id).launch_results == [
        {"status": "passed", "phase": "prefill"},
        {"status": "passed", "phase": "decode"},
    ]


def test_default_cuda_seed_launcher_is_skip_safe_when_smoke_fails(monkeypatch):
    module = _load_serving_shim_example()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvcc"],
            returncode=1,
            stdout="RuntimeError: prepare_callable failed with code -1",
        )

    monkeypatch.setattr(module.shutil, "which", lambda name: "nvcc")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_synthetic_serving_request(
        prompt="hello",
        max_new_tokens=1,
    )

    assert result["status"] == "skipped"
    assert result["text"] == "N"
    assert result["token_ids"] == [1]
    assert result["launch_results"][0]["status"] == "skipped"
    assert "prepare_callable failed with code -1" in result["launch_results"][0]["reason"]


def test_default_cuda_seed_launcher_normalizes_cuda_seed_pass_status(monkeypatch):
    module = _load_serving_shim_example()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvcc"],
            returncode=0,
            stdout=json.dumps({"status": "pass", "runtime": "host_schedule"}),
        )

    monkeypatch.setattr(module.shutil, "which", lambda name: "nvcc")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    launch = module.default_cuda_seed_launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=16,
            block_dim=16,
            arch="compute_90",
        )
    )
    result = module.run_synthetic_serving_request(
        prompt="hello",
        max_new_tokens=2,
    )

    assert launch["status"] == "passed"
    assert launch["cuda_seed"]["status"] == "pass"
    assert result["status"] == "passed"
    assert [item["status"] for item in result["launch_results"]] == ["passed", "passed"]


def test_openai_completion_fixture_uses_synthetic_serving_request_shape():
    module = _load_serving_shim_example()
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    response = module.run_synthetic_openai_completion(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=2,
        device_id=1,
        kernel_launcher=record_launch,
    )

    assert response["id"] == "cmpl-synthetic-0"
    assert response["object"] == "text_completion"
    assert response["created"] == 0
    assert response["model"] == "synthetic-simpler-nv"
    assert response["choices"] == [
        {"index": 0, "text": "NV", "finish_reason": "length"}
    ]
    assert response["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert response["pto_backend"] == "simpler-nv"
    assert response["pto_status"] == "passed"
    assert [launch.phase for launch in launches] == ["prefill", "decode"]


def test_openai_completion_cli_mode_outputs_completion_json(monkeypatch, capsys):
    module = _load_serving_shim_example()

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--openai-completion",
            "--model",
            "synthetic-simpler-nv",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["object"] == "text_completion"
    assert output["choices"][0]["text"] == "N"
    assert output["pto_status"] == "passed"


def test_synthetic_pypto_engine_initializes_model_and_generates_text():
    module = _load_serving_shim_example()
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    engine = module.SyntheticPyptoServingEngine(
        device_id=2,
        kernel_launcher=record_launch,
    )
    engine.init_model("synthetic-simpler-nv", model_dir="synthetic://simpler-nv")

    result = engine.generate(
        model_id="synthetic-simpler-nv",
        prompt="hello",
        max_new_tokens=2,
    )

    assert engine.model_ids() == ["synthetic-simpler-nv"]
    assert result["engine"] == "SyntheticPyptoServingEngine"
    assert result["model_dir"] == "synthetic://simpler-nv"
    assert result["text"] == "NV"
    assert result["token_ids"] == [1, 2]
    assert result["status"] == "passed"
    assert result["launch_count"] == 2
    assert [launch.phase for launch in launches] == ["prefill", "decode"]
    assert {launch.device_id for launch in launches} == {2}


def test_engine_completion_uses_initialized_model_and_openai_shape():
    module = _load_serving_shim_example()

    def record_launch(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    engine = module.SyntheticPyptoServingEngine(kernel_launcher=record_launch)
    engine.init_model("synthetic-simpler-nv", model_dir="synthetic://simpler-nv")

    response = engine.create_openai_completion(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
    )

    assert response["object"] == "text_completion"
    assert response["model"] == "synthetic-simpler-nv"
    assert response["choices"][0]["text"] == "N"
    assert response["usage"]["completion_tokens"] == 1
    assert response["pto_status"] == "passed"
    assert response["pto_engine"] == "SyntheticPyptoServingEngine"


def test_engine_chat_completion_uses_messages_and_openai_shape():
    module = _load_serving_shim_example()
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    engine = module.SyntheticPyptoServingEngine(kernel_launcher=record_launch)
    engine.init_model("synthetic-simpler-nv", model_dir="synthetic://simpler-nv")

    response = engine.create_openai_chat_completion(
        model="synthetic-simpler-nv",
        messages=[
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=2,
    )

    assert response["id"] == "chatcmpl-synthetic-0"
    assert response["object"] == "chat.completion"
    assert response["created"] == 0
    assert response["model"] == "synthetic-simpler-nv"
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "NV"},
            "finish_reason": "length",
        }
    ]
    assert response["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert response["pto_backend"] == "simpler-nv"
    assert response["pto_status"] == "passed"
    assert response["pto_launch_count"] == 2
    assert response["pto_engine"] == "SyntheticPyptoServingEngine"
    assert [launch.phase for launch in launches] == ["prefill", "decode"]


def test_engine_chat_completion_requires_user_content_message():
    module = _load_serving_shim_example()
    engine = module.SyntheticPyptoServingEngine(
        kernel_launcher=lambda request: {"status": "passed", "phase": request.phase}
    )
    engine.init_model("synthetic-simpler-nv")

    with pytest.raises(ValueError, match="at least one user message"):
        engine.create_openai_chat_completion(
            model="synthetic-simpler-nv",
            messages=[{"role": "assistant", "content": "hello"}],
            max_tokens=1,
        )


def test_engine_cli_mode_outputs_engine_result(monkeypatch, capsys):
    module = _load_serving_shim_example()

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--engine",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["engine"] == "SyntheticPyptoServingEngine"
    assert output["text"] == "N"
    assert output["status"] == "passed"


def test_openai_chat_completion_cli_mode_outputs_chat_json(monkeypatch, capsys):
    module = _load_serving_shim_example()

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--openai-chat-completion",
            "--model",
            "synthetic-simpler-nv",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["object"] == "chat.completion"
    assert output["choices"][0]["message"] == {"role": "assistant", "content": "N"}
    assert output["pto_status"] == "passed"


def test_synthetic_fastapi_app_serves_health_models_and_completions():
    module = _load_serving_shim_example()
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    app = module.create_synthetic_openai_app(kernel_launcher=record_launch)
    client = fastapi_testclient.TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json() == {
        "object": "list",
        "data": [
            {
                "id": "synthetic-simpler-nv",
                "object": "model",
                "owned_by": "pypto",
            }
        ],
    }

    response = client.post(
        "/v1/completions",
        json={
            "model": "synthetic-simpler-nv",
            "prompt": "hello",
            "max_tokens": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "NV"
    assert body["pto_engine"] == "SyntheticPyptoServingEngine"
    assert body["pto_status"] == "passed"
    assert [launch.phase for launch in launches] == ["prefill", "decode"]


def test_synthetic_fastapi_app_serves_chat_completions():
    module = _load_serving_shim_example()
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    app = module.create_synthetic_openai_app(kernel_launcher=record_launch)
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "synthetic-simpler-nv",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "NV"}
    assert body["pto_engine"] == "SyntheticPyptoServingEngine"
    assert body["pto_status"] == "passed"
    assert [launch.phase for launch in launches] == ["prefill", "decode"]


def test_http_fixture_cli_mode_outputs_completion_json(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--http-fixture",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["route"] == "/v1/completions"
    assert output["response"]["object"] == "text_completion"
    assert output["response"]["choices"][0]["text"] == "N"
    assert output["response"]["pto_status"] == "passed"


def test_http_fixture_cli_mode_outputs_chat_completion_json(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--http-fixture",
            "--openai-chat-completion",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["route"] == "/v1/chat/completions"
    assert output["response"]["object"] == "chat.completion"
    assert output["response"]["choices"][0]["message"] == {
        "role": "assistant",
        "content": "N",
    }
    assert output["response"]["pto_status"] == "passed"


def test_http_fixture_cli_respects_require_cuda_when_seed_skips(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")

    def fake_launcher(request):
        return {"status": "skipped", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--http-fixture",
            "--require-cuda",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 2
    assert output["status"] == "passed"
    assert output["response"]["pto_status"] == "skipped"


def test_http_fixture_skips_when_testclient_runtime_dependency_is_missing(monkeypatch):
    module = _load_serving_shim_example()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastapi.testclient":
            raise RuntimeError("The starlette.testclient module requires httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = module.run_synthetic_http_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
    )

    assert result["status"] == "skipped"
    assert result["route"] == "/v1/completions"
    assert "requires httpx" in result["reason"]


def test_pypto_serving_source_server_contract_uses_real_routes():
    module = _load_serving_shim_example()
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def record_launch(request):
        launches.append(request)
        return {"status": "passed", "phase": request.phase, "op": request.op}

    app, adapter = module.create_pypto_serving_source_app(
        kernel_launcher=record_launch,
    )
    client = fastapi_testclient.TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "synthetic-simpler-nv"

    response = client.post(
        "/v1/completions",
        json={
            "model": "synthetic-simpler-nv",
            "prompt": "hello",
            "max_tokens": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "synthetic-simpler-nv"
    assert body["choices"][0]["text"] == "NV"
    assert body["choices"][0]["finish_reason"] == "length"
    assert adapter.last_pto_status == "passed"
    assert adapter.last_token_ids == [1, 2]
    assert [launch.phase for launch in launches] == ["prefill", "decode"]


def test_pypto_serving_source_cli_mode_outputs_contract_json(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["server"] == "pypto-serving-source"
    assert output["route"] == "/v1/completions"
    assert output["response"]["object"] == "text_completion"
    assert output["response"]["choices"][0]["text"] == "N"
    assert output["pto_status"] == "passed"


def test_pypto_serving_source_stream_fixture_uses_real_completion_route(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    result = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=2,
    )

    assert result["server"] == "pypto-serving-source"
    assert result["route"] == "/v1/completions"
    assert result["stream"] is True
    assert result["status_code"] == 200
    assert result["event_count"] == 3
    assert result["chunk_count"] == 2
    assert result["done_seen"] is True
    assert result["assembled_text"] == "NV"
    assert result["finish_reason"] == "length"
    assert result["pto_status"] == "passed"
    assert result["pto_token_ids"] == [1, 2]
    assert result["pto_launch_count"] == 2


def test_pypto_serving_source_stream_completion_cli_outputs_summary_json(
    monkeypatch, capsys
):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--pypto-serving-source-stream",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["server"] == "pypto-serving-source"
    assert output["route"] == "/v1/completions"
    assert output["stream"] is True
    assert output["chunk_count"] == 1
    assert output["done_seen"] is True
    assert output["assembled_text"] == "N"
    assert output["finish_reason"] == "length"
    assert output["pto_status"] == "passed"


def test_pypto_serving_source_chat_fixture_uses_real_chat_route(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    result = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=2,
    )

    assert result["server"] == "pypto-serving-source"
    assert result["route"] == "/v1/chat/completions"
    assert result["status_code"] == 200
    assert result["response"]["object"] == "chat.completion"
    choice = result["response"]["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "NV"}
    assert choice["finish_reason"] == "length"
    assert result["pto_status"] == "passed"
    assert result["pto_launch_count"] == 2


def test_pypto_serving_source_stream_chat_fixture_uses_real_chat_route(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    result = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=2,
    )

    assert result["server"] == "pypto-serving-source"
    assert result["route"] == "/v1/chat/completions"
    assert result["stream"] is True
    assert result["status_code"] == 200
    assert result["event_count"] == 3
    assert result["chunk_count"] == 2
    assert result["done_seen"] is True
    assert result["assistant_deltas"] == ["N", "V"]
    assert result["assembled_assistant_text"] == "NV"
    assert result["finish_reason"] == "length"
    assert result["pto_status"] == "passed"
    assert result["pto_token_ids"] == [1, 2]
    assert result["pto_launch_count"] == 2


def test_pypto_serving_source_stream_chat_cli_outputs_summary_json(
    monkeypatch, capsys
):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--pypto-serving-source-chat-stream",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["server"] == "pypto-serving-source"
    assert output["route"] == "/v1/chat/completions"
    assert output["stream"] is True
    assert output["chunk_count"] == 1
    assert output["done_seen"] is True
    assert output["assistant_deltas"] == ["N"]
    assert output["assembled_assistant_text"] == "N"
    assert output["finish_reason"] == "length"
    assert output["pto_status"] == "passed"


def test_pypto_serving_source_chat_cli_mode_outputs_contract_json(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--pypto-serving-source-chat",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["server"] == "pypto-serving-source"
    assert output["route"] == "/v1/chat/completions"
    assert output["response"]["object"] == "chat.completion"
    assert output["response"]["choices"][0]["message"] == {
        "role": "assistant",
        "content": "N",
    }
    assert output["pto_status"] == "passed"


def test_pypto_serving_source_chat_route_fixture_is_documented():
    module = _load_serving_shim_example()
    source_contract = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend"
        / "pypto_serving_source_contract_h200.md"
    ).read_text(encoding="utf-8")

    assert "Source Chat Contract" in source_contract
    assert "run_pypto_serving_source_chat_completion_fixture(...)" in source_contract
    assert "run_pypto_serving_source_stream_completion_fixture(...)" in source_contract
    assert "run_pypto_serving_source_stream_chat_completion_fixture(...)" in source_contract
    assert "--pypto-serving-source-chat" in source_contract
    assert "--pypto-serving-source-stream" in source_contract
    assert "--pypto-serving-source-chat-stream" in source_contract
    assert "/v1/chat/completions" in source_contract
    assert "Current Chat Source Limitation" not in source_contract
    assert hasattr(module, "run_pypto_serving_source_chat_completion_fixture")
    assert hasattr(module, "run_pypto_serving_source_stream_completion_fixture")
    assert hasattr(module, "run_pypto_serving_source_stream_chat_completion_fixture")
