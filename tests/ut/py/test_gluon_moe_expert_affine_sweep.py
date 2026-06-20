import importlib.util
import json
import sys
from pathlib import Path


def _load_gluon_moe_expert_example():
    module_path = "examples/cuda/gluon_moe_expert_affine.py"
    spec = importlib.util.spec_from_file_location(
        "gluon_moe_expert_affine_sweep_example",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gluon_moe_expert_sweep_aggregates_skip_cases(tmp_path, monkeypatch):
    example = _load_gluon_moe_expert_example()
    monkeypatch.chdir(tmp_path)

    result = example.run_moe_expert_sweep(
        output_dir=Path("tmp/gluon-moe-expert-sweep"),
        arch="compute_90",
        skip_reason=lambda: "torch.cuda is not available",
    )

    assert result["schema_version"] == 1
    assert result["kernel_name"] == "moe_expert_affine_f32"
    assert result["status"] == "skipped"
    assert result["case_count"] == len(example.MOE_EXPERT_SWEEP_CASES)
    assert result["passed_cases"] == 0
    assert result["failed_cases"] == 0
    assert result["skipped_cases"] == len(example.MOE_EXPERT_SWEEP_CASES)

    case = result["cases"][0]
    assert case["case_index"] == 0
    assert case["case_name"]
    assert case["shape"]["n"] == example.MOE_EXPERT_SWEEP_CASES[0]["n"]
    assert case["scalars"]["scale_a"] == example.MOE_EXPERT_SWEEP_CASES[0]["scale_a"]
    assert case["scalars"]["scale_b"] == example.MOE_EXPERT_SWEEP_CASES[0]["scale_b"]
    assert case["tolerance"] == {"atol": 1e-06, "rtol": 1e-06}
    assert case["status"] == "skipped"
    assert case["reason"] == "torch.cuda is not available"
    assert case["artifact"]["source_sha256"]
    assert not Path(case["artifact"]["source_path"]).is_absolute()
    assert str(tmp_path) not in json.dumps(result)


def test_gluon_moe_expert_sweep_status_counts_pass_fail_and_skip(monkeypatch):
    example = _load_gluon_moe_expert_example()
    cases = [
        {"name": "passes", "n": 8, "scale_a": 1.0, "scale_b": 0.0, "seed": 0},
        {"name": "fails", "n": 16, "scale_a": -1.0, "scale_b": 0.5, "seed": 1},
        {"name": "skips", "n": 32, "scale_a": 0.25, "scale_b": 2.0, "seed": 2},
    ]

    def fake_run_moe_expert_correctness(**kwargs):
        n = kwargs["n"]
        base = {
            "schema_version": 1,
            "kernel_name": "moe_expert_affine_f32",
            "artifact": {
                "source_path": f"tmp/case-{n}/moe_expert_affine_f32.gluon.py",
                "manifest_path": f"tmp/case-{n}/moe_expert_affine_f32.gluon.json",
                "source_sha256": f"digest-{n}",
                "arch": kwargs["arch"],
                "compiler_role": "pto-isa-replacement",
                "tile_shape": [1, 1, 1],
            },
            "shape": {"n": n},
            "scalars": {
                "scale_a": kwargs["scale_a"],
                "scale_b": kwargs["scale_b"],
            },
            "tolerance": {"atol": kwargs["atol"], "rtol": kwargs["rtol"]},
        }
        if n == 8:
            return {**base, "status": "passed", "max_abs_error": 0.0}
        if n == 16:
            return {**base, "status": "failed", "max_abs_error": 2e-06}
        return {**base, "status": "skipped", "reason": "missing CUDA"}

    monkeypatch.setattr(
        example,
        "run_moe_expert_correctness",
        fake_run_moe_expert_correctness,
    )

    result = example.run_moe_expert_sweep(
        output_dir=Path("tmp/gluon-moe-expert-sweep"),
        arch="compute_90",
        cases=cases,
    )

    assert result["status"] == "failed"
    assert result["case_count"] == 3
    assert result["passed_cases"] == 1
    assert result["failed_cases"] == 1
    assert result["skipped_cases"] == 1
    assert [case["case_name"] for case in result["cases"]] == [
        "passes",
        "fails",
        "skips",
    ]
    assert result["cases"][1]["max_abs_error"] == 2e-06


def test_gluon_moe_expert_sweep_cli_requires_cuda_on_aggregate_skip(
    tmp_path,
    capsys,
    monkeypatch,
):
    example = _load_gluon_moe_expert_example()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(example, "moe_expert_skip_reason", lambda: "missing CUDA")

    code = example.main(
        [
            "--output-dir",
            "tmp/gluon-moe-expert-sweep",
            "--arch",
            "compute_90",
            "--sweep",
            "--require-cuda",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "skipped"
    assert payload["skipped_cases"] == len(example.MOE_EXPERT_SWEEP_CASES)
