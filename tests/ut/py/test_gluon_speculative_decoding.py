import importlib.util
import json
import sys
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler


def _load_gluon_speculative_decoding_example():
    module_path = "examples/cuda/gluon_speculative_decoding.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_speculative_decoding_example",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_cpu_golden():
    return {
        "accepted_token_ids": [[10, 11, 12, 13], [20, -1, -1, -1]],
        "accept_mask": [[1, 1, 1, 1], [1, 0, 0, 0]],
        "accepted_counts": [4, 1],
    }


def test_generate_gluon_speculative_accept_writes_source_and_manifest(tmp_path):
    artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
        "speculative_accept_f32",
        output_dir=tmp_path,
        arch="compute_90",
        tile_shape=(1, 1, 1),
    )

    source = artifact.source_path.read_text(encoding="utf-8")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.kernel_name == "speculative_accept_f32"
    assert artifact.compiler_role == "pto-isa-replacement"
    assert artifact.arch == "compute_90"
    assert artifact.source_path.name == "speculative_accept_f32.gluon.py"
    assert artifact.manifest_path.name == "speculative_accept_f32.gluon.json"
    assert "def speculative_accept_f32_kernel" in source
    assert "def run_speculative_accept_f32" in source
    assert "draft_token_ids_ptr" in source
    assert "draft_probabilities_ptr" in source
    assert "target_probabilities_ptr" in source
    assert "thresholds_ptr" in source
    assert "accepted_token_ids_ptr" in source
    assert "accept_mask_ptr" in source
    assert "accepted_counts_ptr" in source
    assert manifest["source_kind"] == "triton-gluon-python"
    assert manifest["source_path"] == "speculative_accept_f32.gluon.py"


def test_speculative_accept_cpu_golden_stops_at_first_reject():
    example = _load_gluon_speculative_decoding_example()

    golden = example.compute_speculative_accept_cpu_golden(
        draft_token_ids=[
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        draft_probabilities=[
            [0.50, 0.40, 0.20, 0.10],
            [0.50, 0.50, 0.40, 0.20],
        ],
        target_probabilities=[
            [0.50, 0.40, 0.30, 0.20],
            [0.50, 0.10, 0.80, 0.40],
        ],
        thresholds=[
            [1.00, 0.90, 0.80, 0.70],
            [0.80, 0.30, 0.50, 0.50],
        ],
    )

    assert golden == _expected_cpu_golden()


def test_gluon_speculative_decoding_reports_skip_json_and_relative_artifacts(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_speculative_decoding_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_speculative_decoding_correctness(
        output_dir=Path("tmp/gluon-speculative-decoding-local"),
        arch="compute_90",
        rows=2,
        max_draft=4,
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "speculative_accept_f32"
    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"
    assert result["shape"] == {"rows": 2, "max_draft": 4}
    assert result["dtype"] == {
        "draft_token_ids": "int64",
        "draft_probabilities": "float32",
        "target_probabilities": "float32",
        "thresholds": "float32",
        "accepted_token_ids": "int64",
        "accept_mask": "int64",
        "accepted_counts": "int64",
    }
    assert result["request"] == {
        "sampling_operator": "speculative-decoding-accept-reject",
        "acceptance_rule": (
            "accept while threshold <= min(1.0, target_probability / "
            "draft_probability); stop at first reject per row"
        ),
        "deterministic": True,
        "model_stack": "none",
    }
    assert result["cpu_golden"] == _expected_cpu_golden()
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


def test_gluon_speculative_decoding_reports_passed_validation_with_mock_runner(
    tmp_path,
    monkeypatch,
):
    example = _load_gluon_speculative_decoding_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_speculative_decoding_correctness(
        output_dir=Path("tmp/gluon-speculative-decoding-local"),
        arch="compute_90",
        rows=2,
        max_draft=4,
        skip_reason=lambda: None,
        gpu_runner=(
            lambda draft_token_ids, draft_probabilities, target_probabilities, thresholds, **_: (
                example.compute_speculative_accept_cpu_golden(
                    draft_token_ids=draft_token_ids,
                    draft_probabilities=draft_probabilities,
                    target_probabilities=target_probabilities,
                    thresholds=thresholds,
                )
            )
        ),
    )

    assert result["status"] == "passed"
    assert result["validation"] == {
        "accepted_token_ids_match": True,
        "accept_mask_match": True,
        "accepted_counts_match": True,
    }
    assert result["gpu_result"] == result["cpu_golden"]


def test_gluon_speculative_decoding_validation_rejects_truncated_payloads():
    example = _load_gluon_speculative_decoding_example()
    cpu_golden = _expected_cpu_golden()

    missing_token_id_row = {
        **cpu_golden,
        "accepted_token_ids": [[10, 11, 12, 13]],
    }
    missing_token_id_element = {
        **cpu_golden,
        "accepted_token_ids": [[10, 11, 12], [20, -1, -1, -1]],
    }
    missing_mask_element = {
        **cpu_golden,
        "accept_mask": [[1, 1, 1], [1, 0, 0, 0]],
    }
    missing_count = {
        **cpu_golden,
        "accepted_counts": [4],
    }

    row_validation = example._validate_gpu_result(cpu_golden, missing_token_id_row)
    element_validation = example._validate_gpu_result(
        cpu_golden,
        missing_token_id_element,
    )
    mask_validation = example._validate_gpu_result(cpu_golden, missing_mask_element)
    count_validation = example._validate_gpu_result(cpu_golden, missing_count)

    assert row_validation["accepted_token_ids_match"] is False
    assert element_validation["accepted_token_ids_match"] is False
    assert mask_validation["accept_mask_match"] is False
    assert count_validation["accepted_counts_match"] is False


def test_gluon_speculative_decoding_main_requires_cuda_on_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_speculative_decoding_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "speculative_decoding_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-speculative-decoding-local",
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


def test_gluon_speculative_decoding_rejects_absolute_output_dir(capsys):
    example = _load_gluon_speculative_decoding_example()

    code = example.main(["--output-dir", "/tmp/private-output"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "--output-dir must be repo-relative"
    assert "/tmp/private-output" not in json.dumps(payload)
