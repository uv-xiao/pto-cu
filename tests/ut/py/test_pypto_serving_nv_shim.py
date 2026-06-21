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


def test_synthetic_serving_request_reports_failed_launcher_without_raising():
    module = _load_serving_shim_example()

    def failed_launcher(request):
        return {
            "status": "failed",
            "phase": request.phase,
            "launch_kind": "unit-failure",
            "reason": "prepare_callable failed",
        }

    result = module.run_synthetic_serving_request(
        prompt="hello",
        max_new_tokens=1,
        kernel_launcher=failed_launcher,
    )

    assert result["status"] == "failed"
    assert result["text"] == "N"
    assert result["launch_count"] == 1
    assert result["launch_results"][0]["reason"] == "prepare_callable failed"


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


def test_default_launcher_selection_uses_cuda_seed_and_add_op(monkeypatch, capsys):
    module = _load_serving_shim_example()
    launches = []

    def fake_seed_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "op": request.op,
            "launch_kind": "cuda-seed",
        }

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_seed_launcher)

    code = module.main(["--prompt", "hello", "--max-new-tokens", "1"])
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["status"] == "passed"
    assert [launch.op for launch in launches] == ["add"]
    assert output["launch_results"][0]["launch_kind"] == "cuda-seed"


def test_generated_gluon_moe_launcher_records_review_safe_metadata(monkeypatch):
    module = _load_serving_shim_example()

    def fake_correctness(**kwargs):
        return {
            "schema_version": 1,
            "kernel_name": "moe_expert_affine_f32",
            "artifact": {
                "source_path": "tmp/gluon-moe-expert-local/moe_expert_affine_f32.py",
                "source_sha256": "abc123",
                "manifest_path": "tmp/gluon-moe-expert-local/manifest.json",
            },
            "shape": {"n": kwargs["n"]},
            "scalars": {"scale_a": kwargs["scale_a"], "scale_b": kwargs["scale_b"]},
            "status": "skipped",
            "reason": "torch.cuda is not available",
        }

    monkeypatch.setattr(
        module,
        "run_moe_expert_correctness",
        fake_correctness,
        raising=False,
    )

    launcher = module.create_generated_gluon_moe_launcher(output_dir=Path("tmp/unit-gluon"))
    result = launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=31,
            block_dim=16,
            arch="compute_90",
        )
    )

    assert result["status"] == "skipped"
    assert result["launch_kind"] == "gluon-moe-expert"
    assert result["kernel_name"] == "moe_expert_affine_f32"
    assert result["phase"] == "prefill"
    assert result["shape"] == {"n": 31}
    assert result["artifact"]["source_sha256"] == "abc123"
    assert result["generated_kernel"]["reason"] == "torch.cuda is not available"


def test_generated_gluon_topk_sampling_launcher_records_review_safe_metadata(monkeypatch):
    module = _load_serving_shim_example()

    def fake_correctness(**kwargs):
        return {
            "schema_version": 1,
            "kernel_name": "topk_sampling_f32",
            "artifact": {
                "source_path": "tmp/gluon-topk-sampling-local/topk_sampling_f32.py",
                "source_sha256": "topk123",
                "manifest_path": "tmp/gluon-topk-sampling-local/manifest.json",
            },
            "shape": {
                "rows": kwargs["rows"],
                "vocab": kwargs["vocab"],
                "k": kwargs["k"],
            },
            "status": "passed",
            "validation": {
                "values_shape_match": True,
                "indices_shape_match": True,
                "values_match": True,
                "indices_match": True,
                "max_abs_error": 0.0,
            },
            "non_claims": [
                "not FlashInfer integration evidence",
                "not generated-text or tokenizer-semantics evidence",
            ],
        }

    monkeypatch.setattr(
        module,
        "run_topk_sampling_correctness",
        fake_correctness,
        raising=False,
    )

    launcher = module.create_generated_gluon_topk_sampling_launcher(
        output_dir=Path("tmp/unit-gluon-topk"),
        rows=3,
        vocab=16,
        k=5,
    )
    result = launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=31,
            block_dim=16,
            arch="compute_90",
        )
    )

    assert result["status"] == "passed"
    assert result["launch_kind"] == "gluon-topk-sampling"
    assert result["kernel_name"] == "topk_sampling_f32"
    assert result["phase"] == "prefill"
    assert result["shape"] == {"rows": 3, "vocab": 16, "k": 5}
    assert result["artifact"]["source_sha256"] == "topk123"
    assert result["source_sha256"] == "topk123"
    assert result["validation"]["max_abs_error"] == 0.0
    assert "not FlashInfer integration evidence" in result["non_claims"]
    assert result["generated_kernel"]["validation"]["values_match"] is True


def test_generated_gluon_topp_sampling_launcher_records_review_safe_metadata(monkeypatch):
    module = _load_serving_shim_example()

    def fake_correctness(**kwargs):
        return {
            "schema_version": 1,
            "kernel_name": "topp_sampling_f32",
            "artifact": {
                "source_path": "tmp/gluon-topp-sampling-local/topp_sampling_f32.py",
                "source_sha256": "topp123",
                "manifest_path": "tmp/gluon-topp-sampling-local/manifest.json",
            },
            "shape": {
                "rows": kwargs["rows"],
                "vocab": kwargs["vocab"],
                "max_k": kwargs["max_k"],
            },
            "status": "passed",
            "validation": {
                "values_shape_match": True,
                "indices_shape_match": True,
                "selected_counts_shape_match": True,
                "cumulative_probabilities_shape_match": True,
                "values_match": True,
                "indices_match": True,
                "selected_counts_match": True,
                "cumulative_probabilities_match": True,
                "max_abs_error": 0.0,
                "max_cumulative_probability_error": 0.0,
            },
            "non_claims": [
                "not FlashInfer integration evidence",
                "not generated-text or tokenizer-semantics evidence",
            ],
        }

    monkeypatch.setattr(
        module,
        "run_topp_sampling_correctness",
        fake_correctness,
        raising=False,
    )

    launcher = module.create_generated_gluon_topp_sampling_launcher(
        output_dir=Path("tmp/unit-gluon-topp"),
        rows=3,
        vocab=16,
        max_k=6,
        p=0.80,
    )
    result = launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=31,
            block_dim=16,
            arch="compute_90",
        )
    )

    assert result["status"] == "passed"
    assert result["launch_kind"] == "gluon-topp-sampling"
    assert result["kernel_name"] == "topp_sampling_f32"
    assert result["phase"] == "prefill"
    assert result["shape"] == {"rows": 3, "vocab": 16, "max_k": 6}
    assert result["request"]["p"] == 0.80
    assert result["artifact"]["source_sha256"] == "topp123"
    assert result["source_sha256"] == "topp123"
    assert result["validation"]["max_cumulative_probability_error"] == 0.0
    assert "not FlashInfer integration evidence" in result["non_claims"]
    assert result["generated_kernel"]["validation"]["selected_counts_match"] is True


def test_generated_gluon_minp_sampling_launcher_records_review_safe_metadata(monkeypatch):
    module = _load_serving_shim_example()

    def fake_correctness(**kwargs):
        return {
            "schema_version": 1,
            "kernel_name": "minp_sampling_f32",
            "artifact": {
                "source_path": "tmp/gluon-minp-sampling-local/minp_sampling_f32.py",
                "source_sha256": "minp123",
                "manifest_path": "tmp/gluon-minp-sampling-local/manifest.json",
            },
            "shape": {
                "rows": kwargs["rows"],
                "vocab": kwargs["vocab"],
                "max_k": kwargs["max_k"],
            },
            "request": {
                "sampling_operator": "min-p",
                "min_p": kwargs["min_p"],
            },
            "status": "passed",
            "validation": {
                "values_shape_match": True,
                "indices_shape_match": True,
                "selected_counts_shape_match": True,
                "values_match": True,
                "indices_match": True,
                "selected_counts_match": True,
                "max_abs_error": 0.0,
            },
            "non_claims": [
                "not FlashInfer integration evidence",
                "not generated-text or tokenizer-semantics evidence",
            ],
        }

    monkeypatch.setattr(
        module,
        "run_minp_sampling_correctness",
        fake_correctness,
        raising=False,
    )

    launcher = module.create_generated_gluon_minp_sampling_launcher(
        output_dir=Path("tmp/unit-gluon-minp"),
        rows=3,
        vocab=16,
        max_k=6,
        min_p=0.50,
    )
    result = launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=31,
            block_dim=16,
            arch="compute_90",
        )
    )

    assert result["status"] == "passed"
    assert result["launch_kind"] == "gluon-minp-sampling"
    assert result["kernel_name"] == "minp_sampling_f32"
    assert result["phase"] == "prefill"
    assert result["shape"] == {"rows": 3, "vocab": 16, "max_k": 6}
    assert result["request"]["min_p"] == 0.50
    assert result["artifact"]["source_sha256"] == "minp123"
    assert result["source_sha256"] == "minp123"
    assert result["validation"]["max_abs_error"] == 0.0
    assert "not FlashInfer integration evidence" in result["non_claims"]
    assert result["generated_kernel"]["validation"]["selected_counts_match"] is True


def test_persistent_moe_dispatch_combine_launcher_records_review_safe_metadata(monkeypatch):
    module = _load_serving_shim_example()

    def fake_dispatch_combine(**kwargs):
        return {
            "status": "passed",
            "dag_shape": "graph_descriptor_moe_dispatch_combine",
            "completed_count": 5,
            "max_abs_error": 0.0,
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "gluon_expert_bridge": {
                "func_id": 12,
                "kernel_name": "moe_expert_affine_f32",
                "task_name": "gluon_moe_expert_affine_f32",
                "source_kind": "gluon-persistent-task-body-bridge",
                "source_sha256": "bridge123",
            },
            "task_bodies": [
                {
                    "func_id": 12,
                    "name": "gluon_moe_expert_affine_f32",
                    "source_kind": "gluon-persistent-task-body-bridge",
                    "source_sha256": "bridge123",
                }
            ],
            "artifact": {"source_kind": "generated-dispatch"},
        }

    monkeypatch.setattr(
        module,
        "run_moe_dispatch_combine",
        fake_dispatch_combine,
        raising=False,
    )

    launcher = module.create_persistent_moe_dispatch_combine_launcher()
    result = launcher(
        module.KernelLaunchRequest(
            phase="prefill",
            platform="cuda",
            runtime="host_schedule",
            device_id=0,
            op="add",
            n=31,
            block_dim=16,
            arch="compute_90",
        )
    )

    assert result["status"] == "passed"
    assert result["launch_kind"] == "persistent-moe-dispatch-combine"
    assert result["phase"] == "prefill"
    assert result["dag_shape"] == "graph_descriptor_moe_dispatch_combine"
    assert result["shape"] == {"n": 31}
    assert result["completed_count"] == 5
    assert result["max_abs_error"] == 0.0
    assert result["scheduler_error_summary"] == {"count": 0, "code": 0, "task_id": 0}
    assert result["gluon_expert_bridge"]["source_sha256"] == "bridge123"
    assert result["task_body_digest"] == {
        "func_id": 12,
        "source_sha256": "bridge123",
    }
    assert result["persistent_moe"]["artifact"]["source_kind"] == "generated-dispatch"


def test_persistent_moe_dispatch_combine_launcher_reports_api_exceptions(monkeypatch):
    module = _load_serving_shim_example()

    def fake_dispatch_combine(**_kwargs):
        raise RuntimeError("prepare_callable failed for persistent MoE graph")

    monkeypatch.setattr(
        module,
        "run_moe_dispatch_combine",
        fake_dispatch_combine,
        raising=False,
    )

    launcher = module.create_persistent_moe_dispatch_combine_launcher()
    result = launcher(
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

    assert result["status"] == "failed"
    assert result["launch_kind"] == "persistent-moe-dispatch-combine"
    assert result["dag_shape"] == "graph_descriptor_moe_dispatch_combine"
    assert result["error_type"] == "RuntimeError"
    assert result["reason"] == "prepare_callable failed for persistent MoE graph"


def test_generated_launcher_can_run_through_source_route_fixtures(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def generated_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "launch_kind": "gluon-moe-expert",
            "kernel_name": "moe_expert_affine_f32",
            "shape": {"n": request.n},
        }

    completion = module.run_pypto_serving_source_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=generated_launcher,
    )
    chat = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=generated_launcher,
    )
    stream = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=generated_launcher,
    )
    chat_stream = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=generated_launcher,
    )

    assert [item["pto_status"] for item in [completion, chat, stream, chat_stream]] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert [launch.phase for launch in launches] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
    ]


def test_topk_sampling_launcher_can_run_through_source_route_fixtures(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def fake_topk_correctness(**kwargs):
        launches.append(kwargs)
        return {
            "status": "passed",
            "kernel_name": "topk_sampling_f32",
            "shape": {"rows": kwargs["rows"], "vocab": kwargs["vocab"], "k": kwargs["k"]},
            "artifact": {"source_sha256": "topk123"},
            "validation": {"max_abs_error": 0.0},
            "non_claims": ["not FlashInfer integration evidence"],
        }

    monkeypatch.setattr(
        module,
        "run_topk_sampling_correctness",
        fake_topk_correctness,
        raising=False,
    )
    launcher = module.create_generated_gluon_topk_sampling_launcher(
        rows=3,
        vocab=16,
        k=5,
    )

    completion = module.run_pypto_serving_source_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    stream = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat_stream = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )

    results = [completion, chat, stream, chat_stream]
    assert [item["pto_status"] for item in results] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert len(launches) == 4
    assert {item["rows"] for item in launches} == {3}
    assert [
        result["pto_launch_results"][0]["launch_kind"] for result in results
    ] == [
        "gluon-topk-sampling",
        "gluon-topk-sampling",
        "gluon-topk-sampling",
        "gluon-topk-sampling",
    ]


def test_topp_sampling_launcher_can_run_through_source_route_fixtures(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def fake_topp_correctness(**kwargs):
        launches.append(kwargs)
        return {
            "status": "passed",
            "kernel_name": "topp_sampling_f32",
            "shape": {
                "rows": kwargs["rows"],
                "vocab": kwargs["vocab"],
                "max_k": kwargs["max_k"],
            },
            "request": {"p": kwargs["p"]},
            "artifact": {"source_sha256": "topp123"},
            "validation": {"max_cumulative_probability_error": 0.0},
            "non_claims": ["not FlashInfer integration evidence"],
        }

    monkeypatch.setattr(
        module,
        "run_topp_sampling_correctness",
        fake_topp_correctness,
        raising=False,
    )
    launcher = module.create_generated_gluon_topp_sampling_launcher(
        rows=3,
        vocab=16,
        max_k=6,
        p=0.80,
    )

    completion = module.run_pypto_serving_source_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    stream = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat_stream = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )

    results = [completion, chat, stream, chat_stream]
    assert [item["pto_status"] for item in results] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert len(launches) == 4
    assert {item["rows"] for item in launches} == {3}
    assert {item["p"] for item in launches} == {0.80}
    assert [
        result["pto_launch_results"][0]["launch_kind"] for result in results
    ] == [
        "gluon-topp-sampling",
        "gluon-topp-sampling",
        "gluon-topp-sampling",
        "gluon-topp-sampling",
    ]


def test_minp_sampling_launcher_can_run_through_source_route_fixtures(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def fake_minp_correctness(**kwargs):
        launches.append(kwargs)
        return {
            "status": "passed",
            "kernel_name": "minp_sampling_f32",
            "shape": {
                "rows": kwargs["rows"],
                "vocab": kwargs["vocab"],
                "max_k": kwargs["max_k"],
            },
            "request": {"min_p": kwargs["min_p"]},
            "artifact": {"source_sha256": "minp123"},
            "validation": {"max_abs_error": 0.0},
            "non_claims": ["not FlashInfer integration evidence"],
        }

    monkeypatch.setattr(
        module,
        "run_minp_sampling_correctness",
        fake_minp_correctness,
        raising=False,
    )
    launcher = module.create_generated_gluon_minp_sampling_launcher(
        rows=3,
        vocab=16,
        max_k=6,
        min_p=0.50,
    )

    completion = module.run_pypto_serving_source_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    stream = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat_stream = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )

    results = [completion, chat, stream, chat_stream]
    assert [item["pto_status"] for item in results] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert len(launches) == 4
    assert {item["rows"] for item in launches} == {3}
    assert {item["min_p"] for item in launches} == {0.50}
    assert [
        result["pto_launch_results"][0]["launch_kind"] for result in results
    ] == [
        "gluon-minp-sampling",
        "gluon-minp-sampling",
        "gluon-minp-sampling",
        "gluon-minp-sampling",
    ]


def test_persistent_launcher_can_run_through_source_route_fixtures(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def fake_dispatch_combine(**kwargs):
        launches.append(kwargs)
        return {
            "status": "passed",
            "dag_shape": "graph_descriptor_moe_dispatch_combine",
            "completed_count": 5,
            "max_abs_error": 0.0,
            "device_scheduler_errors": {"count": 0, "code": 0, "task_id": 0},
            "gluon_expert_bridge": {
                "func_id": 12,
                "kernel_name": "moe_expert_affine_f32",
                "task_name": "gluon_moe_expert_affine_f32",
                "source_kind": "gluon-persistent-task-body-bridge",
                "source_sha256": "bridge123",
            },
            "task_bodies": [
                {
                    "func_id": 12,
                    "name": "gluon_moe_expert_affine_f32",
                    "source_kind": "gluon-persistent-task-body-bridge",
                    "source_sha256": "bridge123",
                }
            ],
        }

    monkeypatch.setattr(
        module,
        "run_moe_dispatch_combine",
        fake_dispatch_combine,
        raising=False,
    )
    launcher = module.create_persistent_moe_dispatch_combine_launcher()

    completion = module.run_pypto_serving_source_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat = module.run_pypto_serving_source_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    stream = module.run_pypto_serving_source_stream_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )
    chat_stream = module.run_pypto_serving_source_stream_chat_completion_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=launcher,
    )

    results = [completion, chat, stream, chat_stream]
    assert [item["pto_status"] for item in results] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
    assert len(launches) == 4
    assert {item["n"] for item in launches} == {16}
    assert [
        result["pto_launch_results"][0]["launch_kind"] for result in results
    ] == [
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
    ]


def test_generated_kernel_cli_mode_outputs_launch_metadata(monkeypatch, capsys):
    module = _load_serving_shim_example()
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_generated_launcher(**_kwargs):
        def launcher(request):
            return {
                "status": "passed",
                "phase": request.phase,
                "launch_kind": "gluon-moe-expert",
                "kernel_name": "moe_expert_affine_f32",
                "shape": {"n": request.n},
                "artifact": {"source_sha256": "abc123"},
            }

        return launcher

    monkeypatch.setattr(module, "create_generated_gluon_moe_launcher", fake_generated_launcher)

    code = module.main(
        [
            "--kernel-launcher",
            "gluon-moe-expert",
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["pto_status"] == "passed"
    launch = output["pto_launch_results"][0]
    assert launch["launch_kind"] == "gluon-moe-expert"
    assert launch["kernel_name"] == "moe_expert_affine_f32"
    assert launch["shape"] == {"n": 16}
    assert launch["artifact"]["source_sha256"] == "abc123"


def test_topk_sampling_cli_mode_outputs_launch_metadata(monkeypatch, capsys):
    module = _load_serving_shim_example()
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_topk_launcher(**_kwargs):
        def launcher(request):
            return {
                "status": "passed",
                "phase": request.phase,
                "launch_kind": "gluon-topk-sampling",
                "kernel_name": "topk_sampling_f32",
                "shape": {"rows": 3, "vocab": 16, "k": 5},
                "artifact": {"source_sha256": "topk123"},
                "validation": {"max_abs_error": 0.0},
            }

        return launcher

    monkeypatch.setattr(
        module,
        "create_generated_gluon_topk_sampling_launcher",
        fake_topk_launcher,
        raising=False,
    )

    code = module.main(
        [
            "--kernel-launcher",
            "gluon-topk-sampling",
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["pto_status"] == "passed"
    launch = output["pto_launch_results"][0]
    assert launch["launch_kind"] == "gluon-topk-sampling"
    assert launch["kernel_name"] == "topk_sampling_f32"
    assert launch["shape"] == {"rows": 3, "vocab": 16, "k": 5}
    assert launch["artifact"]["source_sha256"] == "topk123"
    assert launch["validation"]["max_abs_error"] == 0.0


def test_topp_sampling_cli_mode_outputs_launch_metadata(monkeypatch, capsys):
    module = _load_serving_shim_example()
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_topp_launcher(**_kwargs):
        def launcher(request):
            return {
                "status": "passed",
                "phase": request.phase,
                "launch_kind": "gluon-topp-sampling",
                "kernel_name": "topp_sampling_f32",
                "shape": {"rows": 3, "vocab": 16, "max_k": 6},
                "request": {"p": 0.80},
                "artifact": {"source_sha256": "topp123"},
                "validation": {"max_cumulative_probability_error": 0.0},
            }

        return launcher

    monkeypatch.setattr(
        module,
        "create_generated_gluon_topp_sampling_launcher",
        fake_topp_launcher,
        raising=False,
    )

    code = module.main(
        [
            "--kernel-launcher",
            "gluon-topp-sampling",
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["pto_status"] == "passed"
    launch = output["pto_launch_results"][0]
    assert launch["launch_kind"] == "gluon-topp-sampling"
    assert launch["kernel_name"] == "topp_sampling_f32"
    assert launch["shape"] == {"rows": 3, "vocab": 16, "max_k": 6}
    assert launch["request"]["p"] == 0.80
    assert launch["artifact"]["source_sha256"] == "topp123"
    assert launch["validation"]["max_cumulative_probability_error"] == 0.0


def test_minp_sampling_cli_mode_outputs_launch_metadata(monkeypatch, capsys):
    module = _load_serving_shim_example()
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_minp_launcher(**_kwargs):
        def launcher(request):
            return {
                "status": "passed",
                "phase": request.phase,
                "launch_kind": "gluon-minp-sampling",
                "kernel_name": "minp_sampling_f32",
                "shape": {"rows": 3, "vocab": 16, "max_k": 6},
                "request": {"min_p": 0.50},
                "artifact": {"source_sha256": "minp123"},
                "validation": {"max_abs_error": 0.0},
            }

        return launcher

    monkeypatch.setattr(
        module,
        "create_generated_gluon_minp_sampling_launcher",
        fake_minp_launcher,
        raising=False,
    )

    code = module.main(
        [
            "--kernel-launcher",
            "gluon-minp-sampling",
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["pto_status"] == "passed"
    launch = output["pto_launch_results"][0]
    assert launch["launch_kind"] == "gluon-minp-sampling"
    assert launch["kernel_name"] == "minp_sampling_f32"
    assert launch["shape"] == {"rows": 3, "vocab": 16, "max_k": 6}
    assert launch["request"]["min_p"] == 0.50
    assert launch["artifact"]["source_sha256"] == "minp123"
    assert launch["validation"]["max_abs_error"] == 0.0


def test_persistent_moe_cli_mode_outputs_launch_metadata(monkeypatch, capsys):
    module = _load_serving_shim_example()
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_persistent_launcher(**_kwargs):
        def launcher(request):
            return {
                "status": "passed",
                "phase": request.phase,
                "launch_kind": "persistent-moe-dispatch-combine",
                "dag_shape": "graph_descriptor_moe_dispatch_combine",
                "shape": {"n": request.n},
                "completed_count": 5,
                "max_abs_error": 0.0,
                "scheduler_error_summary": {"count": 0, "code": 0, "task_id": 0},
                "gluon_expert_bridge": {"source_sha256": "bridge123"},
                "task_body_digest": {"func_id": 12, "source_sha256": "bridge123"},
            }

        return launcher

    monkeypatch.setattr(
        module,
        "create_persistent_moe_dispatch_combine_launcher",
        fake_persistent_launcher,
        raising=False,
    )

    code = module.main(
        [
            "--kernel-launcher",
            "persistent-moe-dispatch-combine",
            "--pypto-serving-source",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["pto_status"] == "passed"
    launch = output["pto_launch_results"][0]
    assert launch["launch_kind"] == "persistent-moe-dispatch-combine"
    assert launch["dag_shape"] == "graph_descriptor_moe_dispatch_combine"
    assert launch["shape"] == {"n": 16}
    assert launch["completed_count"] == 5
    assert launch["task_body_digest"]["source_sha256"] == "bridge123"


def test_persistent_moe_vllm_compat_cli_uses_route_subprocesses(monkeypatch, capsys):
    module = _load_serving_shim_example()
    route_flags = []

    def unexpected_in_process_aggregate(**_kwargs):
        raise AssertionError("persistent aggregate must isolate source routes")

    def fake_run(args, **_kwargs):
        route_flag = next(item for item in args if item.startswith("--pypto-serving-source"))
        route_flags.append(route_flag)
        route = (
            "/v1/chat/completions"
            if "chat" in route_flag
            else "/v1/completions"
        )
        stream = "stream" in route_flag
        launch = {
            "status": "passed",
            "phase": "prefill",
            "launch_kind": "persistent-moe-dispatch-combine",
            "dag_shape": "graph_descriptor_moe_dispatch_combine",
            "completed_count": 5,
            "max_abs_error": 0.0,
            "scheduler_error_summary": {"count": 0, "code": 0, "task_id": 0},
            "task_body_digest": {"func_id": 12, "source_sha256": "bridge123"},
        }
        result = {
            "status": "passed",
            "server": "pypto-serving-source",
            "route": route,
            "status_code": 200,
            "pto_status": "passed",
            "pto_launch_count": 1,
            "pto_launch_results": [launch],
        }
        if stream:
            result.update(
                {
                    "stream": True,
                    "event_count": 2,
                    "chunk_count": 1,
                    "done_seen": True,
                    "finish_reason": "length",
                }
            )
            if "chat" in route_flag:
                result["assembled_assistant_text"] = "N"
            else:
                result["assembled_text"] = "N"
        elif "chat" in route_flag:
            result["response"] = {
                "object": "chat.completion",
                "model": "synthetic-simpler-nv",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "N"},
                        "finish_reason": "length",
                    }
                ],
            }
        else:
            result["response"] = {
                "object": "text_completion",
                "model": "synthetic-simpler-nv",
                "choices": [{"text": "N", "finish_reason": "length"}],
            }
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(result),
        )

    monkeypatch.setattr(
        module,
        "run_pypto_serving_vllm_compat_fixture",
        unexpected_in_process_aggregate,
    )
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    code = module.main(
        [
            "--pypto-serving-vllm-compat",
            "--kernel-launcher",
            "persistent-moe-dispatch-combine",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert route_flags == [
        "--pypto-serving-source",
        "--pypto-serving-source-chat",
        "--pypto-serving-source-stream",
        "--pypto-serving-source-chat-stream",
    ]
    assert output["status"] == "passed"
    assert [
        fixture["observed"]["pto_launch_results"][0]["launch_kind"]
        for fixture in output["fixtures"]
    ] == [
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
        "persistent-moe-dispatch-combine",
    ]


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


def test_pypto_serving_source_app_does_not_require_torch_for_server_route(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    app, adapter = module.create_pypto_serving_source_app(
        kernel_launcher=lambda request: {"status": "passed", "phase": request.phase}
    )

    assert app.title == "PyPTO Serving"
    assert adapter.last_pto_status == ""


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


def test_pypto_serving_vllm_compat_summary_records_structural_fields(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=2,
    )

    assert summary["status"] == "passed"
    assert summary["server"] == "pypto-serving-source"
    assert summary["comparison_baseline"] == "vllm-openai-compatible-deepseek"
    assert summary["checked_fields"] == [
        "route",
        "http_status_200",
        "model_or_stream_object_shape",
        "choice_text_or_message_delta_presence",
        "finish_reason",
        "usage_presence_when_non_streaming",
        "sse_done_presence_when_streaming",
    ]
    assert summary["non_claims"] == [
        "tokenizer semantics",
        "logprob values",
        "stop-token semantics",
        "production readiness",
        "throughput",
        "latency",
        "real DeepSeek weights",
        "simpler-nv/vLLM kernel integration",
    ]

    by_name = {item["name"]: item for item in summary["fixtures"]}
    assert set(by_name) == {
        "completions",
        "chat_completions",
        "stream_completions",
        "stream_chat_completions",
    }
    assert by_name["completions"]["matches"] == {
        "route": True,
        "http_status_200": True,
        "object": True,
        "model": True,
        "choice_text": True,
        "finish_reason": True,
        "usage": False,
    }
    assert by_name["chat_completions"]["matches"] == {
        "route": True,
        "http_status_200": True,
        "object": True,
        "model": True,
        "message_role": True,
        "message_content": True,
        "finish_reason": True,
        "usage": False,
    }
    assert by_name["stream_completions"]["matches"] == {
        "route": True,
        "http_status_200": True,
        "stream": True,
        "choice_text_delta": True,
        "finish_reason": True,
        "sse_done": True,
    }
    assert by_name["stream_chat_completions"]["matches"] == {
        "route": True,
        "http_status_200": True,
        "stream": True,
        "assistant_delta": True,
        "finish_reason": True,
        "sse_done": True,
    }
    assert by_name["completions"]["observed"]["usage_keys"] == []
    assert by_name["stream_chat_completions"]["observed"][
        "assembled_assistant_text"
    ] == "NV"


def test_pypto_serving_vllm_compat_cli_outputs_summary_json(monkeypatch, capsys):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")

    def fake_launcher(request):
        return {"status": "passed", "phase": request.phase, "op": request.op}

    monkeypatch.setattr(module, "default_cuda_seed_launcher", fake_launcher)

    code = module.main(
        [
            "--pypto-serving-vllm-compat",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["status"] == "passed"
    assert output["fixtures"][0]["name"] == "completions"
    assert output["fixtures"][0]["matches"]["usage"] is False
    assert output["fixtures"][2]["name"] == "stream_completions"
    assert output["fixtures"][2]["matches"]["sse_done"] is True
    assert "real DeepSeek weights" in output["non_claims"]


def test_generated_launcher_can_run_through_vllm_compat_summary(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def generated_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "launch_kind": "gluon-moe-expert",
            "kernel_name": "moe_expert_affine_f32",
            "shape": {"n": request.n},
        }

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=generated_launcher,
    )

    assert summary["status"] == "passed"
    assert [launch.phase for launch in launches] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
    ]
    assert [
        fixture["observed"]["pto_launch_count"] for fixture in summary["fixtures"]
    ] == [1, 1, 1, 1]
    assert [
        fixture["observed"]["pto_status"] for fixture in summary["fixtures"]
    ] == ["passed", "passed", "passed", "passed"]


def test_topk_sampling_launcher_can_run_through_vllm_compat_summary(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def topk_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "launch_kind": "gluon-topk-sampling",
            "kernel_name": "topk_sampling_f32",
            "shape": {"rows": 3, "vocab": 16, "k": 5},
            "validation": {"max_abs_error": 0.0},
        }

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=topk_launcher,
    )

    assert summary["status"] == "passed"
    assert [launch.phase for launch in launches] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
    ]
    assert [
        fixture["observed"]["pto_launch_results"][0]["launch_kind"]
        for fixture in summary["fixtures"]
    ] == [
        "gluon-topk-sampling",
        "gluon-topk-sampling",
        "gluon-topk-sampling",
        "gluon-topk-sampling",
    ]


def test_topp_sampling_launcher_can_run_through_vllm_compat_summary(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def topp_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "launch_kind": "gluon-topp-sampling",
            "kernel_name": "topp_sampling_f32",
            "shape": {"rows": 3, "vocab": 16, "max_k": 6},
            "request": {"p": 0.80},
            "validation": {"max_cumulative_probability_error": 0.0},
        }

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=topp_launcher,
    )

    assert summary["status"] == "passed"
    assert [launch.phase for launch in launches] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
    ]
    assert [
        fixture["observed"]["pto_launch_results"][0]["launch_kind"]
        for fixture in summary["fixtures"]
    ] == [
        "gluon-topp-sampling",
        "gluon-topp-sampling",
        "gluon-topp-sampling",
        "gluon-topp-sampling",
    ]


def test_minp_sampling_launcher_can_run_through_vllm_compat_summary(monkeypatch):
    module = _load_serving_shim_example()
    pytest.importorskip("fastapi.testclient")
    if not module.PYPTO_SERVING_SOURCE.is_dir():
        pytest.skip(f"missing {module.PYPTO_SERVING_SOURCE.relative_to(ROOT)}")
    launches = []

    def minp_launcher(request):
        launches.append(request)
        return {
            "status": "passed",
            "phase": request.phase,
            "launch_kind": "gluon-minp-sampling",
            "kernel_name": "minp_sampling_f32",
            "shape": {"rows": 3, "vocab": 16, "max_k": 6},
            "request": {"min_p": 0.50},
            "validation": {"max_abs_error": 0.0},
        }

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
        kernel_launcher=minp_launcher,
    )

    assert summary["status"] == "passed"
    assert [launch.phase for launch in launches] == [
        "prefill",
        "prefill",
        "prefill",
        "prefill",
    ]
    assert [
        fixture["observed"]["pto_launch_results"][0]["launch_kind"]
        for fixture in summary["fixtures"]
    ] == [
        "gluon-minp-sampling",
        "gluon-minp-sampling",
        "gluon-minp-sampling",
        "gluon-minp-sampling",
    ]


def test_vllm_compat_summary_preserves_per_route_launch_results(monkeypatch):
    module = _load_serving_shim_example()

    launch_result = {
        "status": "passed",
        "phase": "prefill",
        "launch_kind": "persistent-moe-dispatch-combine",
        "dag_shape": "graph_descriptor_moe_dispatch_combine",
        "shape": {"n": 16},
        "completed_count": 5,
        "max_abs_error": 0.0,
        "scheduler_error_summary": {"count": 0, "code": 0, "task_id": 0},
        "task_body_digest": {"func_id": 12, "source_sha256": "bridge123"},
    }

    def non_stream_result(route, response):
        return {
            "status": "passed",
            "server": "pypto-serving-source",
            "route": route,
            "status_code": 200,
            "response": response,
            "pto_status": "passed",
            "pto_launch_count": 1,
            "pto_launch_results": [launch_result],
        }

    def stream_result(route, text_key):
        return {
            "status": "passed",
            "server": "pypto-serving-source",
            "route": route,
            "stream": True,
            "status_code": 200,
            "event_count": 2,
            "chunk_count": 1,
            "done_seen": True,
            text_key: "N",
            "finish_reason": "length",
            "pto_status": "passed",
            "pto_launch_count": 1,
            "pto_launch_results": [launch_result],
        }

    monkeypatch.setattr(
        module,
        "run_pypto_serving_source_completion_fixture",
        lambda **_kwargs: non_stream_result(
            "/v1/completions",
            {
                "object": "text_completion",
                "model": "synthetic-simpler-nv",
                "choices": [{"text": "N", "finish_reason": "length"}],
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "run_pypto_serving_source_chat_completion_fixture",
        lambda **_kwargs: non_stream_result(
            "/v1/chat/completions",
            {
                "object": "chat.completion",
                "model": "synthetic-simpler-nv",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "N"},
                        "finish_reason": "length",
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "run_pypto_serving_source_stream_completion_fixture",
        lambda **_kwargs: stream_result("/v1/completions", "assembled_text"),
    )
    monkeypatch.setattr(
        module,
        "run_pypto_serving_source_stream_chat_completion_fixture",
        lambda **_kwargs: stream_result(
            "/v1/chat/completions",
            "assembled_assistant_text",
        ),
    )

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
    )

    assert summary["status"] == "passed"
    for fixture in summary["fixtures"]:
        observed = fixture["observed"]
        assert observed["pto_launch_count"] == 1
        assert observed["pto_launch_results"][0]["launch_kind"] == (
            "persistent-moe-dispatch-combine"
        )
        assert observed["pto_launch_results"][0]["dag_shape"] == (
            "graph_descriptor_moe_dispatch_combine"
        )
        assert observed["pto_launch_results"][0]["completed_count"] == 5
        assert observed["pto_launch_results"][0]["max_abs_error"] == 0.0
        assert observed["pto_launch_results"][0]["scheduler_error_summary"] == {
            "count": 0,
            "code": 0,
            "task_id": 0,
        }
        assert observed["pto_launch_results"][0]["task_body_digest"] == {
            "func_id": 12,
            "source_sha256": "bridge123",
        }


def test_vllm_compat_summary_stops_after_non_passed_pto_launch(monkeypatch):
    module = _load_serving_shim_example()
    calls = []

    def failed_completion(**_kwargs):
        calls.append("completions")
        return {
            "status": "passed",
            "server": "pypto-serving-source",
            "route": "/v1/completions",
            "status_code": 200,
            "response": {
                "object": "text_completion",
                "model": "synthetic-simpler-nv",
                "choices": [{"text": "N", "finish_reason": "length"}],
            },
            "pto_status": "failed",
            "pto_launch_count": 1,
            "pto_launch_results": [
                {
                    "status": "failed",
                    "phase": "prefill",
                    "launch_kind": "persistent-moe-dispatch-combine",
                    "dag_shape": "graph_descriptor_moe_dispatch_combine",
                    "reason": "prepare_callable failed for persistent MoE graph",
                }
            ],
        }

    def unexpected_route(**_kwargs):
        calls.append("unexpected")
        raise AssertionError("aggregate should stop after a non-passed PTO launch")

    monkeypatch.setattr(module, "run_pypto_serving_source_completion_fixture", failed_completion)
    monkeypatch.setattr(module, "run_pypto_serving_source_chat_completion_fixture", unexpected_route)
    monkeypatch.setattr(module, "run_pypto_serving_source_stream_completion_fixture", unexpected_route)
    monkeypatch.setattr(module, "run_pypto_serving_source_stream_chat_completion_fixture", unexpected_route)

    summary = module.run_pypto_serving_vllm_compat_fixture(
        model="synthetic-simpler-nv",
        prompt="hello",
        max_tokens=1,
    )

    assert calls == ["completions"]
    assert summary["status"] == "failed"
    assert len(summary["fixtures"]) == 1
    assert summary["fixtures"][0]["observed"]["pto_status"] == "failed"
    assert summary["fixtures"][0]["observed"]["pto_launch_results"][0]["reason"] == (
        "prepare_callable failed for persistent MoE graph"
    )


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
    assert "--kernel-launcher persistent-moe-dispatch-combine" in source_contract
    assert "Persistent MoE Dispatch/Combine Launch Contract" in source_contract
    assert "run_moe_dispatch_combine(...)" in source_contract
    assert "/v1/chat/completions" in source_contract
    assert "Current Chat Source Limitation" not in source_contract
    assert hasattr(module, "run_pypto_serving_source_chat_completion_fixture")
    assert hasattr(module, "run_pypto_serving_source_stream_completion_fixture")
    assert hasattr(module, "run_pypto_serving_source_stream_chat_completion_fixture")
    assert hasattr(module, "create_persistent_moe_dispatch_combine_launcher")
