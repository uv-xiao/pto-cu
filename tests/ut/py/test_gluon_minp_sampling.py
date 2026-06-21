import importlib.util
import json
import sys
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler


def _load_gluon_minp_sampling_example():
    module_path = "examples/cuda/gluon_minp_sampling.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_minp_sampling_example",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_gluon_minp_sampling_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "minp_sampling_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "minp_sampling_f32"
    assert artifact.compiler_role == "pto-isa-replacement"
    assert artifact.arch == "compute_90"
    assert artifact.source_path.name == "minp_sampling_f32.gluon.py"
    assert artifact.manifest_path.name == "minp_sampling_f32.gluon.json"
    assert "def minp_sampling_f32_kernel" in source
    assert "def run_minp_sampling_f32" in source
    assert "top_values_ptr" in source
    assert "top_indices_ptr" in source
    assert "selected_counts_ptr" in source
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "minp_sampling_f32.gluon.py"


def test_minp_cpu_golden_selects_values_above_scaled_row_max():
    example = _load_gluon_minp_sampling_example()

    probabilities = [
        [0.05, 0.30, 0.10, 0.20, 0.05, 0.15, 0.05, 0.10],
        [0.25, 0.05, 0.25, 0.05, 0.10, 0.05, 0.15, 0.10],
    ]

    golden = example.compute_minp_cpu_golden(
        probabilities,
        min_p=0.5,
        max_k=5,
    )

    assert golden == {
        "values": [[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]],
        "indices": [[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]],
        "selected_counts": [3, 3],
    }


def test_gluon_minp_sampling_reports_broader_shape_with_mock_runner(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_minp_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_minp_sampling_correctness(
        output_dir=Path("tmp/gluon-minp-sampling-local"),
        arch="compute_90",
        rows=3,
        vocab=16,
        max_k=6,
        min_p=0.50,
        skip_reason=lambda: None,
        gpu_runner=lambda probabilities, min_p, max_k, **_: example.compute_minp_cpu_golden(
            probabilities,
            min_p=min_p,
            max_k=max_k,
        ),
    )

    assert result["status"] == "passed"
    assert result["shape"] == {"rows": 3, "vocab": 16, "max_k": 6}
    assert result["request"]["min_p"] == 0.50
    assert result["cpu_golden"] == {
        "values": [
            [0.3, 0.25, 0.15, 0.15, 0.0, 0.0],
            [0.25, 0.2, 0.15, 0.15, 0.13, 0.0],
            [0.22, 0.18, 0.14, 0.12, 0.11, 0.11],
        ],
        "indices": [
            [0, 2, 1, 3, -1, -1],
            [5, 1, 2, 3, 0, -1],
            [8, 0, 4, 2, 1, 5],
        ],
        "selected_counts": [4, 5, 6],
    }
    assert result["gpu_result"] == result["cpu_golden"]
    assert result["validation"]["values_shape_match"] is True
    assert result["validation"]["indices_shape_match"] is True
    assert result["validation"]["selected_counts_shape_match"] is True


def test_gluon_minp_sampling_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_minp_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_minp_sampling_correctness(
        output_dir=Path("tmp/gluon-minp-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        max_k=5,
        min_p=0.5,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "minp_sampling_f32"
    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["shape"] == {"rows": 2, "vocab": 8, "max_k": 5}
    assert result["dtype"] == "float32"
    assert result["request"] == {
        "sampling_operator": "min-p",
        "probabilities_already_normalized": True,
        "min_p": 0.5,
        "threshold_rule": "probability >= min_p * row_max_probability",
        "deterministic": True,
        "tie_break": "lower token id first",
    }
    assert result["cpu_golden"] == {
        "values": [[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]],
        "indices": [[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]],
        "selected_counts": [3, 3],
    }
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


def test_gluon_minp_sampling_reports_passed_validation_with_mock_runner(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_minp_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_minp_sampling_correctness(
        output_dir=Path("tmp/gluon-minp-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        max_k=5,
        min_p=0.5,
        skip_reason=lambda: None,
        gpu_runner=lambda probabilities, min_p, max_k, **_: example.compute_minp_cpu_golden(
            probabilities,
            min_p=min_p,
            max_k=max_k,
        ),
    )

    assert result["status"] == "passed"
    assert result["validation"] == {
        "values_shape_match": True,
        "indices_shape_match": True,
        "selected_counts_shape_match": True,
        "values_match": True,
        "indices_match": True,
        "selected_counts_match": True,
        "max_abs_error": 0.0,
    }
    assert result["gpu_result"] == result["cpu_golden"]


def test_gluon_minp_sampling_validation_rejects_truncated_payloads():
    example = _load_gluon_minp_sampling_example()
    cpu_golden = {
        "values": [[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]],
        "indices": [[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]],
        "selected_counts": [3, 3],
    }

    missing_value_row = {
        **cpu_golden,
        "values": [[0.3, 0.2, 0.15, 0.0, 0.0]],
    }
    missing_value_element = {
        **cpu_golden,
        "values": [[0.3, 0.2, 0.15, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]],
    }
    missing_index_element = {
        **cpu_golden,
        "indices": [[1, 3, 5, -1], [0, 2, 6, -1, -1]],
    }
    missing_selected_count = {
        **cpu_golden,
        "selected_counts": [3],
    }

    row_validation = example._validate_gpu_result(cpu_golden, missing_value_row)
    element_validation = example._validate_gpu_result(cpu_golden, missing_value_element)
    index_validation = example._validate_gpu_result(cpu_golden, missing_index_element)
    count_validation = example._validate_gpu_result(cpu_golden, missing_selected_count)

    assert row_validation["values_match"] is False
    assert element_validation["values_match"] is False
    assert index_validation["indices_match"] is False
    assert count_validation["selected_counts_match"] is False


def test_gluon_minp_sampling_validation_reports_all_payload_shape_mismatches():
    example = _load_gluon_minp_sampling_example()
    cpu_golden = {
        "values": [[0.3, 0.2, 0.15, 0.0, 0.0], [0.25, 0.25, 0.15, 0.0, 0.0]],
        "indices": [[1, 3, 5, -1, -1], [0, 2, 6, -1, -1]],
        "selected_counts": [3, 3],
    }

    validation = example._validate_gpu_result(
        cpu_golden,
        {
            "values": [[0.3, 0.2, 0.15, 0.0, 0.0]],
            "indices": [[1, 3, 5, -1, -1], [0, 2, 6, -1]],
            "selected_counts": [3],
        },
    )

    assert validation["values_shape_match"] is False
    assert validation["indices_shape_match"] is False
    assert validation["selected_counts_shape_match"] is False
    assert validation["values_match"] is False
    assert validation["indices_match"] is False
    assert validation["selected_counts_match"] is False


def test_gluon_minp_sampling_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_minp_sampling_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "minp_sampling_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-minp-sampling-local",
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


def test_gluon_minp_sampling_rejects_absolute_output_dir(capsys):
    example = _load_gluon_minp_sampling_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)
