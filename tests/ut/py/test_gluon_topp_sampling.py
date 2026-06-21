import importlib.util
import json
import sys
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler


def _load_gluon_topp_sampling_example():
    module_path = "examples/cuda/gluon_topp_sampling.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_topp_sampling_example",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_gluon_topp_sampling_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "topp_sampling_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "topp_sampling_f32"
    assert artifact.compiler_role == "pto-isa-replacement"
    assert artifact.arch == "compute_90"
    assert artifact.source_path.name == "topp_sampling_f32.gluon.py"
    assert artifact.manifest_path.name == "topp_sampling_f32.gluon.json"
    assert "def topp_sampling_f32_kernel" in source
    assert "def run_topp_sampling_f32" in source
    assert "top_values_ptr" in source
    assert "top_indices_ptr" in source
    assert "selected_counts_ptr" in source
    assert "cumulative_probs_ptr" in source
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "topp_sampling_f32.gluon.py"


def test_topp_cpu_golden_uses_smallest_prefix_at_probability_boundary():
    example = _load_gluon_topp_sampling_example()

    probabilities = [
        [0.05, 0.30, 0.10, 0.20, 0.05, 0.15, 0.05, 0.10],
        [0.25, 0.05, 0.25, 0.05, 0.10, 0.05, 0.15, 0.10],
    ]

    golden = example.compute_topp_cpu_golden(
        probabilities,
        p=0.75,
        max_k=5,
    )

    assert golden == {
        "values": [[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]],
        "indices": [[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]],
        "selected_counts": [4, 4],
        "cumulative_probabilities": [0.75, 0.75],
    }


def test_gluon_topp_sampling_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topp_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_topp_sampling_correctness(
        output_dir=Path("tmp/gluon-topp-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        max_k=5,
        p=0.75,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "topp_sampling_f32"
    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["shape"] == {"rows": 2, "vocab": 8, "max_k": 5}
    assert result["dtype"] == "float32"
    assert result["request"] == {
        "sampling_operator": "top-p",
        "probabilities_already_normalized": True,
        "p": 0.75,
        "deterministic": True,
        "tie_break": "lower token id first",
    }
    assert result["cpu_golden"] == {
        "values": [[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]],
        "indices": [[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]],
        "selected_counts": [4, 4],
        "cumulative_probabilities": [0.75, 0.75],
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


def test_gluon_topp_sampling_reports_passed_validation_with_mock_runner(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_topp_sampling_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_topp_sampling_correctness(
        output_dir=Path("tmp/gluon-topp-sampling-local"),
        arch="compute_90",
        rows=2,
        vocab=8,
        max_k=5,
        p=0.75,
        skip_reason=lambda: None,
        gpu_runner=lambda probabilities, p, max_k, **_: example.compute_topp_cpu_golden(
            probabilities,
            p=p,
            max_k=max_k,
        ),
    )

    assert result["status"] == "passed"
    assert result["validation"] == {
        "values_match": True,
        "indices_match": True,
        "selected_counts_match": True,
        "cumulative_probabilities_match": True,
        "max_abs_error": 0.0,
        "max_cumulative_probability_error": 0.0,
    }
    assert result["gpu_result"] == result["cpu_golden"]


def test_gluon_topp_sampling_validation_rejects_truncated_float_payloads():
    example = _load_gluon_topp_sampling_example()
    cpu_golden = {
        "values": [[0.3, 0.2, 0.15, 0.1, 0.0], [0.25, 0.25, 0.15, 0.1, 0.0]],
        "indices": [[1, 3, 5, 2, -1], [0, 2, 6, 4, -1]],
        "selected_counts": [4, 4],
        "cumulative_probabilities": [0.75, 0.75],
    }

    missing_value_row = {
        **cpu_golden,
        "values": [[0.3, 0.2, 0.15, 0.1, 0.0]],
    }
    missing_value_element = {
        **cpu_golden,
        "values": [[0.3, 0.2, 0.15, 0.1], [0.25, 0.25, 0.15, 0.1, 0.0]],
    }
    missing_cumulative_probability = {
        **cpu_golden,
        "cumulative_probabilities": [0.75],
    }

    row_validation = example._validate_gpu_result(cpu_golden, missing_value_row)
    element_validation = example._validate_gpu_result(cpu_golden, missing_value_element)
    cumulative_validation = example._validate_gpu_result(
        cpu_golden,
        missing_cumulative_probability,
    )

    assert row_validation["values_match"] is False
    assert element_validation["values_match"] is False
    assert cumulative_validation["cumulative_probabilities_match"] is False


def test_gluon_topp_sampling_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_topp_sampling_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "topp_sampling_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-topp-sampling-local",
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


def test_gluon_topp_sampling_rejects_absolute_output_dir(capsys):
    example = _load_gluon_topp_sampling_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)
