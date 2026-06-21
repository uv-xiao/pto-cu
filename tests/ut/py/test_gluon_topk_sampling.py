import importlib.util
import json
import sys
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler


def _load_gluon_topk_sampling_example():
    module_path = "examples/cuda/gluon_topk_sampling.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_topk_sampling_example",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_gluon_topk_sampling_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "topk_sampling_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "topk_sampling_f32"
    assert artifact.compiler_role == "pto-isa-replacement"
    assert artifact.arch == "compute_90"
    assert artifact.source_path.name == "topk_sampling_f32.gluon.py"
    assert artifact.manifest_path.name == "topk_sampling_f32.gluon.json"
    assert "def topk_sampling_f32_kernel" in source
    assert "def run_topk_sampling_f32" in source
    assert "top_values_ptr" in source
    assert "top_indices_ptr" in source
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "topk_sampling_f32.gluon.py"


def test_topk_cpu_golden_uses_deterministic_value_then_index_order():
    example = _load_gluon_topk_sampling_example()

    logits = [
        [0.1, 0.9, 0.2, 0.9, -0.4, 0.7, 0.3, 0.8],
        [-1.0, -0.5, 0.0, 0.25, 0.25, 0.1, -0.2, 0.9],
    ]

    golden = example.compute_topk_cpu_golden(logits, k=3)

    assert golden == {
        "values": [[0.9, 0.9, 0.8], [0.9, 0.25, 0.25]],
        "indices": [[1, 3, 7], [7, 3, 4]],
    }


def test_gluon_topk_sampling_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topk_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_topk_sampling_correctness(
        output_dir=Path("tmp/gluon-topk-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        k=3,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "topk_sampling_f32"
    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["shape"] == {"rows": 2, "vocab": 8, "k": 3}
    assert result["dtype"] == "float32"
    assert result["request"] == {
        "sampling_operator": "top-k",
        "deterministic": True,
        "tie_break": "lower token id first",
    }
    assert result["cpu_golden"]["values"] == [[0.9, 0.9, 0.8], [0.9, 0.25, 0.25]]
    assert result["cpu_golden"]["indices"] == [[1, 3, 7], [7, 3, 4]]
    assert result["artifact"]["source_sha256"]
    assert not Path(result["artifact"]["source_path"]).is_absolute()
    assert not Path(result["artifact"]["manifest_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)
    assert result["non_claims"] == [
        "not FlashInfer integration evidence",
        "not vLLM or simpler-nv kernel integration evidence",
        "not DeepSeek serving correctness evidence",
        "not generated-text or tokenizer-semantics evidence",
        "not throughput or latency evidence",
    ]


def test_gluon_topk_sampling_supports_broader_shape_fixture(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topk_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_topk_sampling_correctness(
        output_dir=Path("tmp/gluon-topk-shape-coverage-local"),
        arch="compute_90",
        rows=3,
        vocab=16,
        k=5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["status"] == "skipped"
    assert result["shape"] == {"rows": 3, "vocab": 16, "k": 5}
    assert result["cpu_golden"]["values"] == [
        [2.0, 2.0, 1.5, 1.5, 1.2],
        [3.0, 3.0, 2.0, 2.0, 1.5],
        [4.0, 4.0, 3.5, 3.5, 3.5],
    ]
    assert result["cpu_golden"]["indices"] == [
        [7, 11, 2, 3, 9],
        [5, 7, 6, 9, 11],
        [3, 4, 6, 7, 15],
    ]


def test_gluon_topk_sampling_reports_passed_validation_with_mock_runner(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topk_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_topk_sampling_correctness(
        output_dir=Path("tmp/gluon-topk-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        k=3,
        skip_reason=lambda: None,
        gpu_runner=lambda logits, k, **_: example.compute_topk_cpu_golden(logits, k=k),
    )

    assert result["status"] == "passed"
    assert result["validation"] == {
        "values_shape_match": True,
        "indices_shape_match": True,
        "values_match": True,
        "indices_match": True,
        "max_abs_error": 0.0,
    }
    assert result["gpu_result"] == result["cpu_golden"]


def test_gluon_topk_sampling_rejects_truncated_gpu_values_payload(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topk_sampling_example()
    monkeypatch.chdir(tmp_path)

    def truncated_values_runner(logits, k, **_):
        golden = example.compute_topk_cpu_golden(logits, k=k)
        return {
            "values": [row[:-1] for row in golden["values"]],
            "indices": golden["indices"],
        }

    result = example.run_topk_sampling_correctness(
        output_dir=Path("tmp/gluon-topk-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        k=3,
        skip_reason=lambda: None,
        gpu_runner=truncated_values_runner,
    )

    assert result["status"] == "failed"
    assert result["validation"]["values_shape_match"] is False
    assert result["validation"]["indices_shape_match"] is True
    assert result["validation"]["values_match"] is False
    assert result["validation"]["indices_match"] is True


def test_gluon_topk_sampling_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_topk_sampling_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "topk_sampling_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-topk-sampling-local",
            "--arch",
            "compute_90",
            "--require-cuda",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert captured.err == ""
    assert payload["status"] == "skipped"
    assert payload["reason"] == "missing CUDA"


def test_gluon_topk_sampling_rejects_absolute_output_dir(capsys):
    example = _load_gluon_topk_sampling_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)
