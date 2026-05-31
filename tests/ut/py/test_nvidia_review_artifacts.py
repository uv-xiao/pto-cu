import json
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"


def test_nvidia_review_guard_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/check_nvidia_review_ready.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_benchmark_viewer_schema_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_benchmark_viewer_data.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_probe_machine_status_must_match_raw_artifact(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    artifact = tmp_path / "tmp" / "probes" / "a100-probe.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "paper_baseline_id": baseline_id,
                        "status": "pass",
                        "blocking_gaps": [],
                    }
                    for baseline_id in [
                        "mpk",
                        "vdcores",
                        "vllm",
                        "sglang",
                        "thunderkittens",
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )
    def probe_record(baseline_id, checks):
        return {
            "id": f"{baseline_id}_source_entrypoints",
            "paper_baseline_id": baseline_id,
            "title": f"{baseline_id} source entrypoints",
            "latest_status": "pass",
            "latest_artifact_root": "tmp/probes/",
            "latest_machine_status": [
                {
                    "gpu": "A100",
                    "status": "pass",
                    "artifact": "tmp/probes/a100-probe.json",
                    "blocking_gaps": [],
                },
                {
                    "gpu": "H200",
                    "status": "pass",
                    "artifact": "tmp/probes/a100-probe.json",
                    "blocking_gaps": [],
                },
            ],
            "checks": checks,
            "next_action": "fixture",
        }

    data = {
        "paper_baseline_probes": [
            probe_record(
                "mpk",
                [
                    {
                        "kind": "python_module",
                        "module": "transformers",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "vdcores",
                [
                    {
                        "kind": "python_module",
                        "module": "transformers",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "vllm",
                [
                    {
                        "kind": "python_module",
                        "module": "vllm",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "sglang",
                [
                    {
                        "kind": "python_module",
                        "module": "sglang",
                        "why": "fixture",
                    }
                ],
            ),
            probe_record(
                "thunderkittens",
                [
                    {
                        "kind": "python_module",
                        "module": module_name,
                        "why": "fixture",
                    }
                    for module_name in [
                        "torch",
                        "pybind11",
                        "numpy",
                        "pandas",
                        "matplotlib",
                        "tqdm",
                    ]
                ],
            ),
        ]
    }

    baseline_ids = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    module.validate_paper_baseline_probes(data, baseline_ids, tmp_path)
    data["paper_baseline_probes"][0]["latest_machine_status"][1][
        "status"
    ] = "partial"
    try:
        module.validate_paper_baseline_probes(data, baseline_ids, tmp_path)
    except SystemExit as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("machine status drift was not rejected")


def test_imported_paper_baseline_run_rejects_missing_expected_artifact(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    (tmp_path / "tmp" / "paper").mkdir(parents=True)
    (tmp_path / "tmp" / "paper" / "present.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    def run_record(run_id, baseline_id, paper_evaluation_id, serving_ids=None):
        return {
            "id": run_id,
            "paper_baseline_id": baseline_id,
            "paper_evaluation_id": paper_evaluation_id,
            "title": run_id,
            "status": "planned_not_run",
            "hardware_targets": ["H200"],
            "serving_workload_ids": serving_ids or [],
            "workload": {
                "model": "fixture",
                "input_policy": "fixture",
                "output_policy": "fixture",
                "batch_or_concurrency": "fixture",
            },
            "setup_commands": ["true"],
            "run_commands": ["true"],
            "expected_artifacts": ["tmp/paper/future.json"],
            "required_metrics": [
                "correctness",
                "raw_artifacts",
                "model_and_prompt_shape",
                "batch_or_concurrency_policy",
            ],
            "import_target": {
                "viewer_file": (
                    "docs/nvidia-backend/benchmark-viewer/data/results.json"
                ),
                "result_kind": "paper_baseline_result_record",
                "notes": "fixture",
            },
        }

    imported = run_record(
        "mpk_qwen3_native_vs_persistent",
        "mpk",
        "llm_serving_paper_baselines",
        ["mpk_offline_decode"],
    )
    imported["status"] = "imported_to_viewer"
    imported["expected_artifacts"] = [
        "tmp/paper/present.json",
        "tmp/paper/missing.json",
    ]
    data = {
        "paper_baseline_runs": [
            imported,
            run_record(
                "vdcores_llama_decode_correctness",
                "vdcores",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
            ),
            run_record(
                "mpk_persistent_scheduler_trace",
                "mpk",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vdcores_resource_policy_trace",
                "vdcores",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vllm_serving_and_throughput",
                "vllm",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
            ),
            run_record(
                "sglang_serving_and_offline",
                "sglang",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
            ),
            run_record(
                "thunderkittens_tile_kernel",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_full_sweep",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_decode_attention_tile",
                "thunderkittens",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
            ),
        ]
    }

    try:
        module.validate_paper_baseline_runs(
            data,
            {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"},
            {
                "llm_serving_paper_baselines",
                "persistent_device_scheduler_overhead",
                "tensor_core_tile_baselines",
            },
            {"mpk_offline_decode", "vdcores_offline_decode"},
            root=tmp_path,
        )
    except SystemExit as exc:
        assert "expected artifact path missing" in str(exc)
    else:
        raise AssertionError("missing expected artifact was accepted")


def test_llm_serving_paper_baseline_run_requires_shape_and_concurrency(tmp_path):
    script_path = ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def run_record(
        run_id,
        baseline_id,
        paper_evaluation_id,
        serving_ids=None,
        required_metrics=None,
    ):
        return {
            "id": run_id,
            "paper_baseline_id": baseline_id,
            "paper_evaluation_id": paper_evaluation_id,
            "title": run_id,
            "status": "planned_not_run",
            "hardware_targets": ["H200"],
            "serving_workload_ids": serving_ids or [],
            "workload": {
                "model": "fixture",
                "input_policy": "fixture",
                "output_policy": "fixture",
                "batch_or_concurrency": "fixture",
            },
            "setup_commands": ["true"],
            "run_commands": ["true"],
            "expected_artifacts": ["tmp/paper/future.json"],
            "required_metrics": required_metrics
            or ["correctness", "raw_artifacts"],
            "import_target": {
                "viewer_file": (
                    "docs/nvidia-backend/benchmark-viewer/data/results.json"
                ),
                "result_kind": "paper_baseline_result_record",
                "notes": "fixture",
            },
        }

    serving_metrics = [
        "correctness",
        "raw_artifacts",
        "model_and_prompt_shape",
        "batch_or_concurrency_policy",
    ]
    data = {
        "paper_baseline_runs": [
            run_record(
                "mpk_qwen3_native_vs_persistent",
                "mpk",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                ["correctness", "raw_artifacts", "model_and_prompt_shape"],
            ),
            run_record(
                "vdcores_llama_decode_correctness",
                "vdcores",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "mpk_persistent_scheduler_trace",
                "mpk",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vdcores_resource_policy_trace",
                "vdcores",
                "persistent_device_scheduler_overhead",
            ),
            run_record(
                "vllm_serving_and_throughput",
                "vllm",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "sglang_serving_and_offline",
                "sglang",
                "llm_serving_paper_baselines",
                ["mpk_offline_decode"],
                serving_metrics,
            ),
            run_record(
                "thunderkittens_tile_kernel",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_full_sweep",
                "thunderkittens",
                "tensor_core_tile_baselines",
            ),
            run_record(
                "thunderkittens_decode_attention_tile",
                "thunderkittens",
                "llm_serving_paper_baselines",
                ["vdcores_offline_decode"],
                serving_metrics,
            ),
        ]
    }

    try:
        module.validate_paper_baseline_runs(
            data,
            {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"},
            {
                "llm_serving_paper_baselines",
                "persistent_device_scheduler_overhead",
                "tensor_core_tile_baselines",
            },
            {"mpk_offline_decode", "vdcores_offline_decode"},
            root=tmp_path,
        )
    except SystemExit as exc:
        assert "missing LLM serving required metrics" in str(exc)
        assert "batch_or_concurrency_policy" in str(exc)
    else:
        raise AssertionError("LLM serving run without concurrency policy was accepted")


def test_vdcores_scheduler_trace_keeps_diagnostic_scope_separate():
    matrix = json.loads(
        (
            VIEWER_ROOT / "data" / "paper_evaluation_matrix.json"
        ).read_text(encoding="utf-8")
    )
    claim = next(
        item
        for item in matrix["paper_evaluation_matrix"]
        if item["id"] == "persistent_device_scheduler_overhead"
    )
    missing_text = " ".join(claim["missing_evidence"])
    assert "stable baseline instrumentation mode" in missing_text
    assert "non-diagnostic baseline policy" not in missing_text
    assert "measurement scope" in claim["promotion_gate"]

    runs = json.loads(
        (VIEWER_ROOT / "data" / "paper_baseline_runs.json").read_text(
            encoding="utf-8"
        )
    )
    vdcores_run = next(
        item
        for item in runs["paper_baseline_runs"]
        if item["id"] == "vdcores_resource_policy_trace"
    )
    scope = vdcores_run["measurement_scope"]
    assert "without profile-slot diagnostics" in scope["latency_correctness_policy"]
    assert "diagnostic-scope" in scope["scheduler_queue_policy"]
    assert "Do not treat diagnostic scheduler fields" in scope["paper_use_rule"]

    results = json.loads(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8")
    )
    vdcores_result = next(
        item
        for item in results["result_records"]
        if item["benchmark_id"] == "llm_serving_decode"
        and item["method_id"] == "vdcores"
        and item["commit"] == "46872fa4"
    )
    statistic = vdcores_result["statistic"]
    assert statistic["measurement_scope"] == "diagnostic_scheduler_trace"
    trace = statistic["dispatch_trace"]
    assert "tmp-only diagnostics" in trace["trace_scope"]
    assert "stable VDCores baseline build" in trace["paper_use_rule"]
    assert (
        statistic["resource_policy"]["instrumentation_scope"]
        == "diagnostic_profile_slots"
    )


def test_nvidia_changelog_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_nvidia_changelog.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_cuda_example_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_cuda_examples.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_remote_evaluation_validator_passes():
    result = subprocess.run(
        [sys.executable, ".agents/checks/validate_remote_evaluation.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_cuda_viewer_export_generates_contract_records(tmp_path):
    capture = {
        "metadata": {
            "git_commit": "abc1234",
            "label": "fixture-capture",
        },
        "results": [
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 120,
                "device_wall_ns": 80,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 160,
                "device_wall_ns": 100,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_host_schedule",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 200,
                "device_wall_ns": 120,
                "status": "pass",
            },
            {
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm_graph",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 60,
                "device_wall_ns": 40,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "direct_runtime_sgemm",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 90,
                "device_wall_ns": 70,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_persistent_dag_graph_tensor_core",
                "n": 1024,
                "task_count": 4,
                "host_wall_ns": 110,
                "device_wall_ns": 85,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_stream_serial",
                "n": 2,
                "task_count": 1,
                "host_wall_ns": 300,
                "device_wall_ns": 300,
                "status": "pass",
            },
            {
                "machine": "hina",
                "baseline": "pto_stream_parallel",
                "n": 2,
                "task_count": 1,
                "host_wall_ns": 160,
                "device_wall_ns": 160,
                "status": "pass",
            },
        ],
    }
    capture_path = tmp_path / "cuda-benchmark.json"
    output_path = tmp_path / "viewer-records.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py",
            str(capture_path),
            "--artifact-root",
            "tmp/cuda-backend/fixture/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))

    host_record = next(
        record for record in records if record["method_id"] == "pto_host_schedule"
    )
    assert host_record["benchmark_id"] == "host_schedule_vector_ops"
    assert host_record["hardware"]["gpu"] == "A100"
    assert host_record["hardware"]["compute_target"] == "compute_80"
    assert host_record["statistic"]["sample_count"] == 3
    assert host_record["statistic"]["host_wall_ns"] == 160
    assert host_record["statistic"]["device_wall_ns"] == 100
    assert host_record["statistic"]["host_wall_p50_ns"] == 160
    assert host_record["statistic"]["host_wall_p90_ns"] == 192
    assert host_record["statistic"]["host_wall_p99_ns"] == 199
    assert host_record["statistic"]["host_wall_mean_ns"] == 160
    assert host_record["statistic"]["host_wall_stdev_ns"] == 40
    assert host_record["statistic"]["host_wall_min_ns"] == 120
    assert host_record["statistic"]["host_wall_max_ns"] == 200
    assert host_record["statistic"]["device_wall_p50_ns"] == 100
    assert host_record["statistic"]["device_wall_p90_ns"] == 116
    assert host_record["statistic"]["device_wall_p99_ns"] == 119
    assert host_record["statistic"]["device_wall_mean_ns"] == 100
    assert host_record["statistic"]["device_wall_stdev_ns"] == 20
    assert host_record["statistic"]["device_wall_min_ns"] == 80
    assert host_record["statistic"]["device_wall_max_ns"] == 120
    assert host_record["raw_artifact"] == "tmp/cuda-backend/fixture/"
    assert host_record["correctness"] == "pass"

    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "cublas_sgemm_graph"
        and record["hardware"]["gpu"] == "H200"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "direct_runtime"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["dtype"] == "float32 naive SGEMM"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "pto_persistent_device"
        and record["hardware"]["gpu"] == "A100"
        and record["inputs"]["dtype"] == "tf32 WMMA tensor-core, f32 accumulator"
        and record["statistic"]["sample_count"] == 1
        for record in records
    )
    assert any(
        record["benchmark_id"] == "host_schedule_stream_concurrency"
        and record["method_id"] == "pto_stream_serial"
        and record["inputs"]["shape"] == "two independent n=1 vector kernels"
        for record in records
    )
    assert any(
        record["benchmark_id"] == "host_schedule_stream_concurrency"
        and record["method_id"] == "pto_stream_parallel"
        and record["inputs"]["shape"] == "two independent n=1 vector kernels"
        for record in records
    )


def test_triton_tensor_tile_capture_exports_fixture_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
            "source": "fixture",
        },
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "fixture-driver",
            "cuda_toolkit": "fixture-cuda",
            "clock_policy": "fixture-clock",
        },
        "inputs": {
            "shape": "n=1024, tensor tile 16x16x16",
            "dtype": "tf32 Triton tl.dot, f32 accumulator",
            "repeat_policy": "3-repeat fixture capture",
        },
        "samples": [
            {"host_wall_ns": 1000, "device_wall_ns": 700, "max_abs_error": 0.0},
            {"host_wall_ns": 1200, "device_wall_ns": 800, "max_abs_error": 0.0},
            {"host_wall_ns": 1400, "device_wall_ns": 900, "max_abs_error": 0.0},
        ],
    }
    raw_path = tmp_path / "triton-capture.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py",
            "--input-json",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/triton/fixture/",
            "--viewer-output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["benchmark_id"] == "tensor_core_tile"
    assert record["method_id"] == "triton"
    assert record["hardware"]["gpu"] == "A100"
    assert record["inputs"]["shape"] == "n=1024, tensor tile 16x16x16"
    assert record["statistic"]["sample_count"] == 3
    assert record["statistic"]["host_wall_ns"] == 1200
    assert record["statistic"]["device_wall_ns"] == 800
    assert record["statistic"]["max_abs_error"] == 0.0
    assert record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/triton/fixture/"
    assert record["correctness"] == "pass"


def test_cutlass_tensor_tile_capture_exports_fixture_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
            "source": "fixture",
            "cutlass_commit": "1732ed7",
        },
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "fixture-driver",
            "cuda_toolkit": "fixture-cuda",
            "clock_policy": "fixture-clock",
        },
        "inputs": {
            "shape": "n=1024, tensor tile 16x16x16",
            "dtype": "tf32 CUTLASS Gemm tensor op, f32 accumulator",
            "repeat_policy": "3-repeat fixture capture",
        },
        "samples": [
            {"host_wall_ns": 2000, "device_wall_ns": 1200, "max_abs_error": 0.0},
            {"host_wall_ns": 2200, "device_wall_ns": 1300, "max_abs_error": 0.0},
            {"host_wall_ns": 2400, "device_wall_ns": 1400, "max_abs_error": 0.0},
        ],
    }
    raw_path = tmp_path / "cutlass-capture.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/cutlass_tensor_tile_capture.py",
            "--input-json",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/cutlass/fixture/",
            "--viewer-output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["benchmark_id"] == "tensor_core_tile"
    assert record["method_id"] == "cutlass"
    assert record["hardware"]["gpu"] == "A100"
    assert record["inputs"]["shape"] == "n=1024, tensor tile 16x16x16"
    assert record["statistic"]["sample_count"] == 3
    assert record["statistic"]["host_wall_ns"] == 2200
    assert record["statistic"]["device_wall_ns"] == 1300
    assert record["statistic"]["max_abs_error"] == 0.0
    assert record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/cutlass/fixture/"
    assert record["correctness"] == "pass"


def test_paper_baseline_viewer_export_generates_contract_records(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "vllm_serving_and_throughput",
                "benchmark_id": "llm_serving_decode",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                    "driver": "570.86.15",
                    "cuda_toolkit": "12.8",
                    "clock_policy": "application clocks locked",
                },
                "inputs": {
                    "shape": "model=fixture,prompt_tokens=128,decode_tokens=32",
                    "dtype": "bfloat16",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_capture",
                    "sample_count": 3,
                    "end_to_end_latency_ns": 1000000,
                    "time_to_first_token_ns": 250000,
                    "inter_token_latency_ns": 50000,
                    "throughput_tokens_per_s": 640.0,
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "paper-baseline.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/vllm/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))

    assert records == [
        {
            "benchmark_id": "llm_serving_decode",
            "method_id": "vllm",
            "hardware": {
                "gpu": "H200",
                "machine": "dasys-h200x8",
                "compute_target": "compute_90",
                "driver": "570.86.15",
                "cuda_toolkit": "12.8",
                "clock_policy": "application clocks locked",
            },
            "commit": "abc1234",
            "inputs": {
                "shape": "model=fixture,prompt_tokens=128,decode_tokens=32",
                "dtype": "bfloat16",
                "repeat_policy": "warmup=1,repeat=3",
            },
            "statistic": {
                "kind": "paper_baseline_capture",
                "sample_count": 3,
                "host_wall_ns": 1000000,
                "device_wall_ns": 0,
                "end_to_end_latency_ns": 1000000,
                "time_to_first_token_ns": 250000,
                "inter_token_latency_ns": 50000,
                "throughput_tokens_per_s": 640.0,
            },
            "raw_artifact": "tmp/cuda-backend/paper-baselines/vllm/",
            "correctness": "pass",
        }
    ]


def test_paper_baseline_viewer_export_preserves_scheduler_trace_metadata(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "scheduler_overhead_ns": 42000,
                    "dispatch_trace": {
                        "task_count": 9,
                        "func_ids": [1, 2, 11, 1, 2, 1, 6, 1, 1],
                    },
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                    "task_registry": {
                        "generated_kernel": "mirage_persistent_kernel",
                    },
                },
                "correctness": "pass",
            },
            {
                "paper_baseline_run_id": "vdcores_resource_policy_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, virtual-core resource trace",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 1100000,
                    "device_wall_ns": 850000,
                    "scheduler_overhead_ns": 51000,
                    "dispatch_trace": {
                        "task_count": 9,
                        "queue_ids": ["compute", "memory"],
                    },
                    "queue_pressure": {
                        "ready_queue_max": 4,
                        "memory_queue_max": 2,
                    },
                    "resource_policy": {
                        "virtual_cores": 8,
                        "memory_compute_split": "2:6",
                    },
                },
                "correctness": "pass",
            },
        ],
    }
    raw_path = tmp_path / "scheduler-baseline.json"
    output_path = tmp_path / "viewer-records.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/scheduler-fixture/",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = json.loads(output_path.read_text(encoding="utf-8"))
    by_method = {record["method_id"]: record for record in records}

    mpk_statistic = by_method["mpk"]["statistic"]
    assert mpk_statistic["scheduler_overhead_ns"] == 42000
    assert mpk_statistic["dispatch_trace"]["task_count"] == 9
    assert mpk_statistic["resource_policy"]["scheduler_blocks"] == 3
    assert mpk_statistic["task_registry"]["generated_kernel"]

    vdcores_statistic = by_method["vdcores"]["statistic"]
    assert vdcores_statistic["scheduler_overhead_ns"] == 51000
    assert vdcores_statistic["dispatch_trace"]["queue_ids"] == [
        "compute",
        "memory",
    ]
    assert vdcores_statistic["queue_pressure"]["ready_queue_max"] == 4
    assert vdcores_statistic["resource_policy"]["virtual_cores"] == 8


def test_thunderkittens_capture_builds_serving_decode_result():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_mha_capture.py"
    )
    spec = importlib.util.spec_from_file_location("thunderkittens_mha_capture", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_raw_result_record(
        paper_baseline_run_id="thunderkittens_decode_attention_tile",
        benchmark_id="llm_serving_decode",
        machine="dasys-h200x8",
        cuda_toolkit="12.8",
        clock_policy="not recorded",
        gpu_metadata={
            "gpu": "NVIDIA H200 NVL",
            "driver": "580.126.20",
            "compute_target": "compute_90",
        },
        shape={
            "b": 4,
            "h": 1,
            "n": 256,
            "d": 64,
            "causal": True,
            "dtype": "bfloat16",
        },
        latency={
            "warmup": 5,
            "repeats": 20,
            "sample_count": 20,
            "p50_ns": 32000,
        },
        correctness={"status": "pass"},
        serving_workload_id="vdcores_offline_decode",
        prompt_tokens=128,
        decode_tokens=64,
    )

    assert result["paper_baseline_run_id"] == "thunderkittens_decode_attention_tile"
    assert result["benchmark_id"] == "llm_serving_decode"
    assert result["hardware"]["gpu"] == "H200"
    assert (
        result["inputs"]["shape"]
        == "vdcores_offline_decode,mha_h100,b=4,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64"
    )
    assert result["metrics"]["kind"] == "paper_baseline_serving_tile_capture"
    assert result["metrics"]["end_to_end_latency_ns"] == 32000
    assert result["metrics"]["time_to_first_token_ns"] == 32000
    assert result["metrics"]["inter_token_latency_ns"] == 500
    assert result["metrics"]["throughput_tokens_per_s"] == 8000000
    assert result["correctness"] == "pass"


def test_thunderkittens_full_sweep_capture_builds_importable_record():
    import importlib.util

    script = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_full_sweep_capture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "thunderkittens_full_sweep_capture",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_raw_result_record(
        machine="dasys-h200x8",
        cuda_toolkit="12.8",
        clock_policy="not recorded",
        gpu_metadata={
            "gpu": "NVIDIA H200 NVL",
            "driver": "580.126.20",
            "compute_target": "compute_90",
        },
        shape={"b": 1, "h": 1, "n": 256, "d": 64, "causal": True, "dtype": "bfloat16"},
        latency={
            "warmup": 5,
            "repeats": 20,
            "sample_count": 20,
            "p50_ns": 32000,
        },
        correctness={"status": "pass", "max_abs_diff": 0.001953125},
    )

    assert result["paper_baseline_run_id"] == "thunderkittens_full_sweep"
    assert result["benchmark_id"] == "tensor_core_tile"
    assert result["hardware"]["gpu"] == "H200"
    assert result["inputs"]["shape"] == (
        "mha_h100_full_sweep,b=1,h=1,n=256,d=64,causal=True"
    )
    assert result["metrics"]["kind"] == "paper_baseline_full_sweep_capture"
    assert result["metrics"]["device_wall_ns"] == 32000
    assert result["metrics"]["throughput"] > 0
    assert result["metrics"]["max_abs_error"] == 0.001953125
    assert result["correctness"] == "pass"


def test_paper_baseline_results_update_marks_imported_run(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "scheduler_overhead_ns": 42000,
                    "dispatch_trace": {"task_count": 9},
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "mpk-scheduler.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    results_path = tmp_path / "results.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    audit_path = tmp_path / "paper_readiness_audit.json"
    viewer_path = tmp_path / "viewer-records.json"
    results_path.write_text(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runs_path.write_text(
        (VIEWER_ROOT / "data" / "paper_baseline_runs.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/mpk/fixture/",
            "--results",
            str(results_path),
            "--runs",
            str(runs_path),
            "--viewer-output",
            str(viewer_path),
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    viewer_records = json.loads(viewer_path.read_text(encoding="utf-8"))
    assert len(viewer_records) == 1
    assert viewer_records[0]["method_id"] == "mpk"
    assert viewer_records[0]["statistic"]["scheduler_overhead_ns"] == 42000

    updated_results = json.loads(results_path.read_text(encoding="utf-8"))
    assert any(
        record["method_id"] == "mpk"
        and record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/mpk/fixture/"
        for record in updated_results["result_records"]
    )

    updated_runs = json.loads(runs_path.read_text(encoding="utf-8"))
    run_by_id = {
        record["id"]: record
        for record in updated_runs["paper_baseline_runs"]
    }
    assert run_by_id["mpk_persistent_scheduler_trace"]["status"] == (
        "imported_to_viewer"
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    persistent_claim = {
        record["id"]: record for record in audit["claim_audits"]
    }["persistent_device_scheduler_overhead"]
    mpk_status = [
        record
        for record in persistent_claim["paper_baseline_run_statuses"]
        if record["id"] == "mpk_persistent_scheduler_trace"
    ][0]
    assert mpk_status["status"] == "imported_to_viewer"


def test_paper_baseline_results_update_rejects_missing_required_metric(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "mpk_persistent_scheduler_trace",
                "benchmark_id": "graph_layered_cross",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "n=1024, nine-task layered-cross DAG",
                    "dtype": "float32",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_scheduler_trace",
                    "sample_count": 3,
                    "host_wall_ns": 900000,
                    "device_wall_ns": 700000,
                    "dispatch_trace": {"task_count": 9},
                    "resource_policy": {
                        "scheduler_blocks": 3,
                        "worker_blocks": 4,
                    },
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "missing-scheduler-overhead.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    results_path = tmp_path / "results.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    audit_path = tmp_path / "paper_readiness_audit.json"
    viewer_path = tmp_path / "viewer-records.json"
    results_path.write_text(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runs_path.write_text(
        (VIEWER_ROOT / "data" / "paper_baseline_runs.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/mpk/fixture/",
            "--results",
            str(results_path),
            "--runs",
            str(runs_path),
            "--viewer-output",
            str(viewer_path),
            "--audit-output",
            str(audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "scheduler_overhead" in result.stdout
    assert not viewer_path.exists()
    assert not audit_path.exists()


def test_paper_baseline_run_readiness_probe_exports_run_blockers(tmp_path):
    baseline_root = tmp_path / "baselines"
    mpk_root = baseline_root / "mirage-mpk"
    vdcores_root = baseline_root / "vdcores"
    vllm_root = baseline_root / "vllm"
    sglang_root = baseline_root / "sglang"
    (mpk_root / "demo" / "qwen3").mkdir(parents=True)
    (mpk_root / "demo" / "qwen3" / "demo.py").write_text(
        "print('mpk fixture')\n",
        encoding="utf-8",
    )
    (vdcores_root / "app" / "python" / "llama3").mkdir(parents=True)
    (vdcores_root / "python" / "dae").mkdir(parents=True)
    (vdcores_root / "app" / "python" / "llama3" / "sched.py").write_text(
        "print('vdcores fixture')\n",
        encoding="utf-8",
    )
    vllm_root.mkdir(parents=True)
    sglang_root.mkdir(parents=True)

    baselines = {
        "paper_baselines": [
            {
                "id": "mpk",
                "source": {
                    "local_tmp_path": str(mpk_root),
                },
            },
            {
                "id": "vdcores",
                "source": {
                    "local_tmp_path": str(vdcores_root),
                },
            },
            {
                "id": "vllm",
                "source": {
                    "local_tmp_path": str(vllm_root),
                },
            },
            {
                "id": "sglang",
                "source": {
                    "local_tmp_path": str(sglang_root),
                },
            },
        ]
    }
    runs = {
        "paper_baseline_runs": [
            {
                "id": "mpk_persistent_scheduler_trace",
                "paper_baseline_id": "mpk",
                "title": "MPK Persistent Scheduler Trace",
                "run_commands": [
                    "cd tmp/baselines/mirage-mpk && python demo/qwen3/demo.py --use-mirage --profiling"
                ],
                "model_access": {
                    "requires_hf_token": False,
                    "public_models": ["Qwen/Qwen3-1.7B"],
                },
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json"
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "vdcores_resource_policy_trace",
                "paper_baseline_id": "vdcores",
                "title": "VDCores Resource Policy Trace",
                "run_commands": [
                    "cd tmp/baselines/vdcores && python app/python/llama3/sched.py --correctness"
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/vdcores/resource-policy-trace.json"
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "vllm_serving_and_throughput",
                "paper_baseline_id": "vllm",
                "title": "vLLM Serving And Throughput",
                "status": "planned_not_run",
                "run_commands": [
                    "vllm bench serve --backend vllm --model <model>",
                    "vllm bench throughput --model <model>",
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/vllm/bench-serve.json",
                    "tmp/cuda-backend/paper-baselines/vllm/bench-throughput.json",
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
            {
                "id": "sglang_serving_and_offline",
                "paper_baseline_id": "sglang",
                "title": "SGLang Serving And Offline",
                "status": "planned_not_run",
                "run_commands": [
                    "python -m sglang.bench_serving --backend sglang",
                    "python -m sglang.bench_offline_throughput --model-path <model>",
                ],
                "expected_artifacts": [
                    "tmp/cuda-backend/paper-baselines/sglang/bench-serving.json",
                    "tmp/cuda-backend/paper-baselines/sglang/offline-throughput.json",
                ],
                "required_metrics": ["correctness", "raw_artifacts"],
            },
        ]
    }
    probes = {
        "paper_baseline_probes": [
            {
                "id": "mpk_source_entrypoints",
                "paper_baseline_id": "mpk",
                "latest_status": "pass",
                "latest_machine_status": [],
            },
            {
                "id": "vdcores_source_entrypoints",
                "paper_baseline_id": "vdcores",
                "latest_status": "pass",
                "latest_machine_status": [],
            },
            {
                "id": "vllm_source_entrypoints",
                "paper_baseline_id": "vllm",
                "latest_status": "partial",
                "latest_machine_status": [
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "blocking_gaps": ["python_module failed: vllm"],
                    }
                ],
            },
            {
                "id": "sglang_source_entrypoints",
                "paper_baseline_id": "sglang",
                "latest_status": "partial",
                "latest_machine_status": [
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "blocking_gaps": [
                            "python_import failed: sglang.bench_serving"
                        ],
                    }
                ],
            },
        ]
    }
    baselines_path = tmp_path / "paper_baselines.json"
    runs_path = tmp_path / "paper_baseline_runs.json"
    probes_path = tmp_path / "paper_baseline_probes.json"
    env_plans_path = tmp_path / "paper_baseline_environment_plans.json"
    output_root = tmp_path / "run-readiness"
    viewer_output = tmp_path / "paper_baseline_run_readiness.json"
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")
    runs_path.write_text(json.dumps(runs), encoding="utf-8")
    probes_path.write_text(json.dumps(probes), encoding="utf-8")
    env_plans_path.write_text(
        json.dumps(
            {
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "status": "plan_ready",
                        "raw_artifact": "tmp/cuda-backend/paper-baselines/environment-plans/run-readiness-fixture/environment-plans.json",
                    },
                    {
                        "id": "sglang_runtime_environment",
                        "paper_baseline_id": "sglang",
                        "status": "plan_ready",
                        "raw_artifact": "tmp/cuda-backend/paper-baselines/environment-plans/run-readiness-fixture/environment-plans.json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py",
            "--runs",
            str(runs_path),
            "--baselines",
            str(baselines_path),
            "--probes",
            str(probes_path),
            "--env-plans",
            str(env_plans_path),
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    status = json.loads(viewer_output.read_text(encoding="utf-8"))
    records = {
        item["paper_baseline_run_id"]: item
        for item in status["paper_baseline_run_readiness"]
    }
    assert {
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
    } <= set(records)

    assert records["mpk_persistent_scheduler_trace"]["latest_status"] == "pass"
    assert not any(
        "HF_TOKEN" in gap
        for gap in records["mpk_persistent_scheduler_trace"]["blocking_gaps"]
    )
    assert any(
        check["kind"] == "environment"
        and check["name"] == "public_model_access"
        and check["status"] == "pass"
        for check in records["mpk_persistent_scheduler_trace"]["checks"]
    )
    vdcores = records["vdcores_resource_policy_trace"]
    assert vdcores["latest_status"] == "partial"
    assert any("dae.runtime" in gap for gap in vdcores["blocking_gaps"])
    assert records["vllm_serving_and_throughput"]["latest_status"] == "partial"
    assert any(
        "python_module failed: vllm" in gap
        for gap in records["vllm_serving_and_throughput"]["blocking_gaps"]
    )
    assert records["sglang_serving_and_offline"]["latest_status"] == "partial"
    assert any(
        "sglang.bench_serving" in gap
        for gap in records["sglang_serving_and_offline"]["blocking_gaps"]
    )
    assert any(
        check["kind"] == "environment_plan"
        and check["status"] == "pass"
        for check in records["vllm_serving_and_throughput"]["checks"]
    )
    assert (output_root / "run-readiness.json").is_file()
    artifact = json.loads((output_root / "run-readiness.json").read_text())
    assert artifact["metadata"]["pto_commit"] == "abc1234"


def test_paper_baseline_environment_plan_exports_isolated_serving_envs(tmp_path):
    baseline_root = tmp_path / "baselines"
    vllm_root = baseline_root / "vllm"
    sglang_root = baseline_root / "sglang"
    (vllm_root / "requirements").mkdir(parents=True)
    (vllm_root / "requirements" / "build").mkdir(parents=True)
    (vllm_root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["torch == 2.11.0", "setuptools-rust>=1.9.0", "setuptools-scm>=8.0"]
[project]
name = "vllm"
dependencies = ["uvloop", "pydantic", "cbor2"]
""",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "build" / "cuda.txt").write_text(
        "cmake>=3.26.1\nninja\nsetuptools-rust>=1.9.0\nsetuptools-scm>=8.0\n",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "common.txt").write_text(
        "pydantic >= 2.12.0\ncbor2\nuvloop==0.22.1\n",
        encoding="utf-8",
    )
    (vllm_root / "requirements" / "cuda.txt").write_text(
        "torch==2.11.0\ntorchvision==0.26.0\nflashinfer-python==0.6.11.post2\ntilelang==0.1.9\n",
        encoding="utf-8",
    )
    (sglang_root / "python").mkdir(parents=True)
    (sglang_root / "python" / "pyproject.toml").write_text(
        """
[project]
name = "sglang"
dependencies = [
  "sglang",
  "torch==2.11.0",
  "torchvision",
  "orjson",
  "uvloop",
  "flashinfer_python[cu13]==0.6.11.post1",
  "tilelang==0.1.8",
]
""",
        encoding="utf-8",
    )
    baselines = {
        "paper_baselines": [
            {
                "id": "vllm",
                "source": {
                    "local_tmp_path": str(vllm_root),
                    "commit": "1234567890abcdef1234567890abcdef12345678",
                },
            },
            {
                "id": "sglang",
                "source": {
                    "local_tmp_path": str(sglang_root),
                    "commit": "abcdef1234567890abcdef1234567890abcdef12",
                },
            },
        ]
    }
    baselines_path = tmp_path / "paper_baselines.json"
    output_root = tmp_path / "environment-plans"
    viewer_output = tmp_path / "paper_baseline_environment_plans.json"
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py",
            "--baselines",
            str(baselines_path),
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    records = {
        item["paper_baseline_id"]: item
        for item in payload["paper_baseline_environment_plans"]
    }
    assert set(records) == {"vllm", "sglang"}
    assert records["vllm"]["status"] == "plan_ready"
    assert records["sglang"]["status"] == "plan_ready"
    assert records["vllm"]["environment_path"].startswith(
        "tmp/cuda-backend/paper-baselines/envs/vllm-12345678"
    )
    assert records["sglang"]["environment_path"].startswith(
        "tmp/cuda-backend/paper-baselines/envs/sglang-abcdef12"
    )
    assert all(
        "--system-site-packages" in command
        for command in [
            records["vllm"]["install_commands"][0],
            records["sglang"]["install_commands"][0],
        ]
    )
    assert all(
        command.startswith("REPO_ROOT=$PWD && cd ")
        for command in [
            command
            for command in (
                records["vllm"]["install_commands"]
                + records["sglang"]["install_commands"]
            )
            if command.startswith("REPO_ROOT=") or command.startswith("cd ")
        ]
    )
    assert ".venv" not in " ".join(
        records["vllm"]["install_commands"] + records["sglang"]["install_commands"]
    )
    assert not any(
        command.startswith("python -m pip")
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
    )
    assert all(
        "PYTHONNOUSERSITE=1" in command
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
        if "-m pip install" in command
    )
    assert all(
        "PATH=" in command and "/bin:$PATH" in command
        for command in (
            records["vllm"]["install_commands"]
            + records["sglang"]["install_commands"]
        )
        if "-m pip install" in command
    )
    assert "--user" not in " ".join(
        records["vllm"]["install_commands"] + records["sglang"]["install_commands"]
    )
    assert "PYTHONNOUSERSITE=1" in " ".join(
        records["sglang"]["validation_commands"]
    )
    vllm_packages = {
        package["name"]: package
        for package in records["vllm"]["critical_packages"]
    }
    assert vllm_packages["torch"]["declared"]
    assert vllm_packages["setuptools-rust"]["declared"]
    assert vllm_packages["setuptools-scm"]["declared"]
    assert vllm_packages["cbor2"]["declared"]
    assert any(
        "requirements/build/cuda.txt" in command
        for command in records["vllm"]["install_commands"]
    )
    assert records["vllm"]["preflight_after_install_steps"] == 5
    assert len(records["vllm"]["preflight_commands"]) == 1
    assert "vllm_spinloop_preflight.py" in records["vllm"]["preflight_commands"][0]
    assert records["sglang"]["preflight_commands"] == []
    assert (output_root / "environment-plans.json").is_file()


def test_vllm_spinloop_preflight_rejects_python310_limited_api(tmp_path):
    source = tmp_path / "vllm"
    (source / "csrc").mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        """
define_extension_target(
  spinloop
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES csrc/spinloop.cpp
  USE_SABI 3.11
  WITH_SOABI)
""",
        encoding="utf-8",
    )
    (source / "csrc" / "spinloop.cpp").write_text(
        "void f() { Py_buffer buffer; PyBuffer_Release(&buffer); }\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '[3, 10, 12]\\n'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/vllm_spinloop_preflight.py",
            "--source",
            str(source),
            "--env-python",
            str(fake_python),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert "Py_buffer" in payload["blocker"]
    assert "Python 3.10.12" in payload["blocker"]


def test_paper_baseline_environment_attempt_captures_bounded_steps(tmp_path):
    env_path = tmp_path / "tmp" / "envs" / "vllm-fixture"
    marker = env_path / "marker.txt"
    plans_path = tmp_path / "paper_baseline_environment_plans.json"
    viewer_output = tmp_path / "paper_baseline_environment_attempts.json"
    output_root = tmp_path / "attempts"
    plans_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "pto_commit": "abc1234",
                    "artifact_root": "tmp/environment-plans/",
                    "source_files": [
                        "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json"
                    ],
                },
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "title": "vLLM fixture env",
                        "status": "plan_ready",
                        "source_path": "tmp/baselines/vllm",
                        "source_commit": "1234567890abcdef",
                        "environment_path": str(env_path),
                        "python_policy": "fixture policy",
                        "dependency_sources": ["pyproject.toml"],
                        "critical_packages": [],
                        "manual_packages": [],
                        "install_commands": [
                            f"python3 -c \"import pathlib; pathlib.Path(r'{env_path}').mkdir(parents=True, exist_ok=True)\"",
                            f"python3 -c \"import pathlib; pathlib.Path(r'{marker}').write_text('installed')\"",
                        ],
                        "validation_commands": [
                            f"python3 -c \"import pathlib; assert pathlib.Path(r'{marker}').read_text() == 'installed'\""
                        ],
                        "execution_gaps": ["fixture gap"],
                        "notes": ["fixture note"],
                        "next_action": "fixture next",
                        "raw_artifact": "tmp/environment-plans/environment-plans.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--max-steps",
            "2",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    [attempt] = payload["paper_baseline_environment_attempts"]
    assert attempt["paper_baseline_id"] == "vllm"
    assert attempt["environment_plan_id"] == "vllm_runtime_environment"
    assert attempt["status"] == "partial"
    assert attempt["steps_completed"] == 2
    assert attempt["steps_total"] == 3
    assert attempt["environment_path"] == str(env_path)
    assert all(step["status"] == "pass" for step in attempt["steps"])
    assert all(".venv" not in step["command"] for step in attempt["steps"])
    assert (output_root / "environment-attempt.json").is_file()
    assert (output_root / "step-01.log").is_file()
    assert marker.read_text(encoding="utf-8") == "installed"


def test_paper_baseline_environment_attempt_appends_resume_window(tmp_path):
    env_path = tmp_path / "tmp" / "envs" / "vllm-fixture"
    marker = env_path / "marker.txt"
    plans_path = tmp_path / "paper_baseline_environment_plans.json"
    viewer_output = tmp_path / "paper_baseline_environment_attempts.json"
    first_output_root = tmp_path / "attempts-first"
    second_output_root = tmp_path / "attempts-second"
    plans_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "pto_commit": "abc1234",
                    "artifact_root": "tmp/environment-plans/",
                    "source_files": [
                        "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json"
                    ],
                },
                "paper_baseline_environment_plans": [
                    {
                        "id": "vllm_runtime_environment",
                        "paper_baseline_id": "vllm",
                        "title": "vLLM fixture env",
                        "status": "plan_ready",
                        "source_path": "tmp/baselines/vllm",
                        "source_commit": "1234567890abcdef",
                        "environment_path": str(env_path),
                        "python_policy": "fixture policy",
                        "dependency_sources": ["pyproject.toml"],
                        "critical_packages": [],
                        "manual_packages": [],
                        "install_commands": [
                            f"python3 -c \"import pathlib; pathlib.Path(r'{env_path}').mkdir(parents=True, exist_ok=True)\"",
                            f"python3 -c \"import pathlib; pathlib.Path(r'{marker}').write_text('installed')\"",
                        ],
                        "validation_commands": [
                            f"python3 -c \"import pathlib; assert pathlib.Path(r'{marker}').read_text() == 'installed'\""
                        ],
                        "execution_gaps": ["fixture gap"],
                        "notes": ["fixture note"],
                        "next_action": "fixture next",
                        "raw_artifact": "tmp/environment-plans/environment-plans.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(first_output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--max-steps",
            "2",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert first.returncode == 0, first.stdout

    second = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py",
            "--plans",
            str(plans_path),
            "--baseline",
            "vllm",
            "--output-root",
            str(second_output_root),
            "--viewer-output",
            str(viewer_output),
            "--commit",
            "abc1234",
            "--start-step",
            "3",
            "--max-steps",
            "1",
            "--attempt-id-suffix",
            "step03",
            "--append-viewer",
            "--timeout-seconds",
            "10",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert second.returncode == 0, second.stdout

    payload = json.loads(viewer_output.read_text(encoding="utf-8"))
    attempts = payload["paper_baseline_environment_attempts"]
    assert [attempt["id"] for attempt in attempts] == [
        "vllm_environment_attempt_abc1234",
        "vllm_environment_attempt_abc1234_step03",
    ]
    resume = attempts[1]
    assert resume["start_step"] == 3
    assert resume["end_step"] == 3
    assert resume["steps_total"] == 3
    assert resume["steps_completed"] == 1
    assert resume["steps"][0]["index"] == 3
    assert resume["steps"][0]["kind"] == "validation"
    assert resume["steps"][0]["status"] == "pass"
    assert (second_output_root / "step-03.log").is_file()


def test_paper_baseline_viewer_export_rejects_bool_sample_count(tmp_path):
    raw = {
        "metadata": {
            "pto_commit": "abc1234",
        },
        "results": [
            {
                "paper_baseline_run_id": "vllm_serving_and_throughput",
                "benchmark_id": "llm_serving_decode",
                "hardware": {
                    "gpu": "H200",
                    "machine": "dasys-h200x8",
                    "compute_target": "compute_90",
                },
                "inputs": {
                    "shape": "model=fixture",
                    "dtype": "bfloat16",
                    "repeat_policy": "warmup=1,repeat=3",
                },
                "metrics": {
                    "kind": "paper_baseline_capture",
                    "sample_count": True,
                },
                "correctness": "pass",
            }
        ],
    }
    raw_path = tmp_path / "paper-baseline-invalid.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
            str(raw_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/vllm/",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid sample_count" in result.stdout


def test_paper_baseline_probe_collects_source_readiness(tmp_path):
    baseline_root = tmp_path / "fake-mpk"
    (baseline_root / "demo" / "qwen3").mkdir(parents=True)
    (baseline_root / "demo" / "qwen3" / "demo.py").write_text(
        "print('fixture')\n",
        encoding="utf-8",
    )
    (baseline_root / "python" / "fixture_pkg").mkdir(parents=True)
    (baseline_root / "python" / "fixture_pkg" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init"],
        cwd=baseline_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=baseline_root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=baseline_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=baseline_root,
        text=True,
    ).strip()

    baselines_path = tmp_path / "paper_baselines.json"
    probes_path = tmp_path / "paper_baseline_probes.json"
    output_path = tmp_path / "probe.json"

    baselines = {
        "paper_baselines": [
            {
                "id": "mpk",
                "name": "Mirage Persistent Kernel",
                "paper_role": "fixture",
                "status": "source_cloned_for_survey",
                "source": {
                    "upstream_url": "https://example.invalid/mpk",
                    "local_tmp_path": str(baseline_root),
                    "commit": commit,
                },
                "paper_baselines_to_reproduce": ["fixture"],
                "next_action": "fixture",
            }
        ]
    }
    probes = {
        "paper_baseline_probes": [
            {
                "id": "mpk_source_entrypoints",
                "paper_baseline_id": "mpk",
                "title": "MPK source entrypoints",
                "latest_status": "not_captured",
                "latest_artifact_root": "tmp/cuda-backend/paper-baselines/probes/",
                "latest_machine_status": [
                    {
                        "gpu": "A100",
                        "status": "pass",
                        "artifact": str(output_path),
                        "blocking_gaps": [],
                    },
                    {
                        "gpu": "H200",
                        "status": "partial",
                        "artifact": str(output_path),
                        "blocking_gaps": ["fixture gap"],
                    },
                ],
                "checks": [
                    {
                        "kind": "path_exists",
                        "path": "demo/qwen3/demo.py",
                        "why": "fixture entrypoint",
                    },
                    {
                        "kind": "py_compile",
                        "path": "demo/qwen3/demo.py",
                        "why": "fixture syntax",
                    },
                    {
                        "kind": "python_import",
                        "module": "fixture_pkg",
                        "pythonpath": "python",
                        "python_no_user_site": True,
                        "why": "fixture import through source path",
                    },
                ],
                "next_action": "fixture",
            }
        ]
    }
    baselines_path.write_text(json.dumps(baselines), encoding="utf-8")
    probes_path.write_text(json.dumps(probes), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
            "--baselines",
            str(baselines_path),
            "--probes",
            str(probes_path),
            "--output",
            str(output_path),
            "--artifact-root",
            "tmp/cuda-backend/paper-baselines/probes/fixture/",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["artifact_root"] == (
        "tmp/cuda-backend/paper-baselines/probes/fixture/"
    )
    assert payload["probes"][0]["paper_baseline_id"] == "mpk"
    assert payload["probes"][0]["status"] == "pass"
    assert payload["probes"][0]["source_commit_actual"] == commit
    assert [check["status"] for check in payload["probes"][0]["checks"]] == [
        "pass",
        "pass",
        "pass",
    ]
    import_check = payload["probes"][0]["checks"][2]
    assert import_check["pythonpath"] == "python"
    assert import_check["python_no_user_site"] is True


def test_paper_probe_status_update_materializes_machine_status(tmp_path):
    paired_root = tmp_path / "tmp" / "paired-probe"
    paired_root.mkdir(parents=True)
    for filename, status, gaps in [
        ("a100-probe.json", "pass", []),
        ("h200-probe.json", "partial", ["python_module failed: fixture"]),
    ]:
        (paired_root / filename).write_text(
            json.dumps(
                {
                    "probes": [
                        {
                            "paper_baseline_id": "fixture",
                            "status": status,
                            "blocking_gaps": gaps,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    probes_path = tmp_path / "paper_baseline_probes.json"
    output_path = tmp_path / "updated-probes.json"
    probes_path.write_text(
        json.dumps(
            {
                "paper_baseline_probes": [
                    {
                        "id": "fixture_source_entrypoints",
                        "paper_baseline_id": "fixture",
                        "title": "Fixture Source Entrypoints",
                        "latest_status": "pass",
                        "latest_artifact_root": "tmp/old/",
                        "latest_machine_status": [],
                        "checks": [],
                        "next_action": "fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_probe_status_update.py",
            "--probes",
            str(probes_path),
            "--paired-artifact-root",
            str(paired_root),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    record = payload["paper_baseline_probes"][0]

    assert record["latest_status"] == "partial"
    assert record["latest_artifact_root"].endswith("tmp/paired-probe/")
    assert record["latest_machine_status"] == [
        {
            "gpu": "A100",
            "status": "pass",
            "artifact": record["latest_artifact_root"] + "a100-probe.json",
            "blocking_gaps": [],
        },
        {
            "gpu": "H200",
            "status": "partial",
            "artifact": record["latest_artifact_root"] + "h200-probe.json",
            "blocking_gaps": ["python_module failed: fixture"],
        },
    ]


def test_paper_readiness_audit_matches_current_viewer_data(tmp_path):
    output_path = tmp_path / "paper-readiness-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_readiness_audit.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = json.loads(
        (VIEWER_ROOT / "data" / "paper_readiness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == committed
    assert committed["overall_status"] == "not_paper_ready"
    assert committed["ready_claims"] == 1
    assert committed["blocked_claims"] == 3

    by_id = {claim["id"]: claim for claim in committed["claim_audits"]}
    host_claim = by_id["host_schedule_launch_overhead"]
    assert host_claim["ready_for_paper_claim"]
    assert host_claim["blockers"] == []
    assert host_claim["next_actions"] == []
    llm_claim = by_id["llm_serving_paper_baselines"]
    assert not llm_claim["ready_for_paper_claim"]
    assert any(
        "mpk_qwen3_native_vs_persistent is planned_not_run" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert any(
        "Readiness probe for sglang is partial" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert any(
        "Run readiness vllm_serving_and_throughput is partial" in blocker
        for blocker in llm_claim["blockers"]
    )
    assert any(
        action["source"] == "run_readiness"
        and action["paper_baseline_run_id"] == "vllm_serving_and_throughput"
        and "Resolve blocking gaps" in action["action"]
        for action in llm_claim["next_actions"]
    )
    assert any(
        action["source"] == "probe"
        and action["paper_baseline_id"] == "sglang"
        and "Install SGLang runtime dependencies" in action["action"]
        for action in llm_claim["next_actions"]
    )
    llm_readiness_ids = {
        item["paper_baseline_run_id"]
        for item in llm_claim["paper_baseline_run_readiness_statuses"]
    }
    assert {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
    } <= llm_readiness_ids
    assert "thunderkittens_decode_attention_tile" not in llm_readiness_ids
    assert not any(
        "No paper baseline run record is attached to this claim for thunderkittens"
        in blocker
        for blocker in llm_claim["blockers"]
    )
    persistent_claim = by_id["persistent_device_scheduler_overhead"]
    persistent_run_ids = {
        run["id"] for run in persistent_claim["paper_baseline_run_statuses"]
    }
    assert {
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
    } <= persistent_run_ids
    persistent_readiness = {
        item["paper_baseline_run_id"]: item
        for item in persistent_claim["paper_baseline_run_readiness_statuses"]
    }
    persistent_attempts = {
        item["paper_baseline_run_id"]: item
        for item in persistent_claim["execution_attempt_statuses"]
    }
    mpk_run_status = next(
        run
        for run in persistent_claim["paper_baseline_run_statuses"]
        if run["id"] == "mpk_persistent_scheduler_trace"
    )
    assert mpk_run_status["status"] == "imported_to_viewer"
    assert "mpk_persistent_scheduler_trace" not in persistent_attempts
    assert "vdcores_resource_policy_trace" not in persistent_attempts
    vdcores_run_status = next(
        run
        for run in persistent_claim["paper_baseline_run_statuses"]
        if run["id"] == "vdcores_resource_policy_trace"
    )
    assert vdcores_run_status["status"] == "imported_to_viewer"
    assert not any(
        "VDCores queue-pressure and scheduler-overhead metadata"
        in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert any(
        "stable baseline instrumentation mode" in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        "Latest execution attempt "
        "mpk_qwen3_0p6b_profile_termination_diagnostic_h200"
        in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        action["source"] == "execution_attempt"
        and action["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        for action in persistent_claim["next_actions"]
    )
    assert not any(
        "Readiness probe for mpk is partial" in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        action["source"] == "run_readiness"
        and action["paper_baseline_run_id"] == "vdcores_resource_policy_trace"
        for action in persistent_claim["next_actions"]
    )
    assert "mpk_persistent_scheduler_trace" not in persistent_readiness
    assert "vdcores_resource_policy_trace" not in persistent_readiness
    assert not any(
        "Scheduler-overhead breakdown" in blocker
        for blocker in persistent_claim["blockers"]
    )
    assert not any(
        "No paper baseline run record is attached" in blocker
        for blocker in persistent_claim["blockers"]
    )
    tensor_claim = by_id["tensor_core_tile_baselines"]
    tensor_run_ids = {
        run["id"] for run in tensor_claim["paper_baseline_run_statuses"]
    }
    assert {
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
    } <= tensor_run_ids
    assert any(
        "Resolve remaining official ThunderKittens upstream sweep gaps"
        in blocker
        and "non-MHA ThunderKittens kernels" in blocker
        for blocker in tensor_claim["blockers"]
    )
    assert not any(
        "thunderkittens_full_sweep is planned_not_run" in blocker
        for blocker in tensor_claim["blockers"]
    )
    assert any(
        action["source"] == "matrix_missing_evidence"
        and "FlashAttention 3 bindings are unavailable" in action["action"]
        for action in tensor_claim["next_actions"]
    )


def test_paper_readiness_work_queue_matches_current_audit(tmp_path):
    output_path = tmp_path / "work-queue.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_readiness_work_queue.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = json.loads(
        (VIEWER_ROOT / "data" / "paper_readiness_work_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == committed
    assert committed["overall_status"] == "not_paper_ready"
    assert committed["summary"]["total_work_items"] == 8
    assert committed["summary"]["work_items_by_source"] == {
        "matrix_missing_evidence": 3,
        "probe": 2,
        "run_readiness": 3,
    }
    work_items = committed["work_items"]
    assert all(not item["ready_for_paper_claim"] for item in work_items)
    assert any(
        item["claim_id"] == "llm_serving_paper_baselines"
        and item["source"] == "probe"
        and item["paper_baseline_id"] == "sglang"
        and "Install SGLang runtime dependencies" in item["action"]
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        and item["source"] == "run_readiness"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_run_id"] == "mpk_persistent_scheduler_trace"
        for item in work_items
    )
    assert not any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "execution_attempt"
        and item["paper_baseline_run_id"] == "vdcores_resource_policy_trace"
        for item in work_items
    )
    assert any(
        item["claim_id"] == "persistent_device_scheduler_overhead"
        and item["source"] == "matrix_missing_evidence"
        and "stable baseline instrumentation mode" in item["action"]
        for item in work_items
    )


def test_nvidia_goal_progress_matches_current_artifacts(tmp_path):
    output_path = tmp_path / "goal-progress.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "nvidia_goal_progress.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    generated = json.loads(output_path.read_text(encoding="utf-8"))
    committed = json.loads(
        (VIEWER_ROOT / "data" / "goal_progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated == committed
    assert committed["overall_status"] == "in_progress"
    assert committed["summary"]["criteria_total"] == 8
    assert committed["summary"]["criteria_met"] >= 6
    assert committed["summary"]["criteria_in_progress"] >= 1
    by_id = {item["id"]: item for item in committed["acceptance_criteria"]}
    assert by_id["paper_grade_results"]["status"] == "in_progress"
    assert by_id["paper_grade_results"]["blocking_work_items"] == 8
    assert by_id["paper_grade_results"]["paper_readiness_status"] == (
        "not_paper_ready"
    )
    assert by_id["remote_evaluation"]["status"] == "met"
    assert by_id["benchmark_viewer"]["status"] == "met"
    assert all(item["evidence_refs"] for item in committed["acceptance_criteria"])
    assert all(item["verification"] for item in committed["acceptance_criteria"])


def test_nvidia_review_artifact_refresh_regenerates_all_generated_json(tmp_path):
    output_dir = tmp_path / "viewer-data"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "refresh_nvidia_review_artifacts.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for filename in [
        "paper_readiness_audit.json",
        "paper_readiness_work_queue.json",
        "goal_progress.json",
    ]:
        generated = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        committed = json.loads(
            (VIEWER_ROOT / "data" / filename).read_text(encoding="utf-8")
        )
        assert generated == committed
    assert "paper_readiness_audit.json" in result.stdout
    assert "paper_readiness_work_queue.json" in result.stdout
    assert "goal_progress.json" in result.stdout


def test_paper_serving_command_plan_generates_policy_commands(tmp_path):
    output_path = tmp_path / "serving-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/cuda-backend-eval/scripts/"
            "paper_serving_command_plan.py",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    records = payload["serving_command_plans"]

    assert payload["metadata"]["model_tier"] == "primary"
    assert len(records) == 35
    by_id = {record["id"]: record for record in records}
    vllm_mpk = by_id[
        "vllm_serving_and_throughput:mpk_offline_decode:batch16"
    ]
    assert vllm_mpk["model"] == "Qwen/Qwen3-8B"
    assert vllm_mpk["prompt_tokens"] == 64
    assert vllm_mpk["decode_tokens"] == 1024
    assert vllm_mpk["batch_size"] == 16
    assert any(
        "--max-concurrency 16" in command["command"]
        and "--input-len 64" in command["command"]
        and "--output-len 1024" in command["command"]
        for command in vllm_mpk["commands"]
    )

    sglang_vdcores = by_id[
        "sglang_serving_and_offline:vdcores_offline_decode:batch8"
    ]
    assert sglang_vdcores["prompt_tokens"] == 128
    assert sglang_vdcores["decode_tokens"] == 64
    assert any(
        "--random-input-len 128" in command["command"]
        and "--random-output-len 64" in command["command"]
        for command in sglang_vdcores["commands"]
    )
    assert all(
        command["command"].startswith(
            "env PYTHONPATH=$PWD/tmp/baselines/sglang/python:$PYTHONPATH "
        )
        for command in sglang_vdcores["commands"]
    )
    thunderkittens_decode = by_id[
        "thunderkittens_decode_attention_tile:vdcores_offline_decode:batch4"
    ]
    assert thunderkittens_decode["prompt_tokens"] == 128
    assert thunderkittens_decode["decode_tokens"] == 64
    assert any(
        command["kind"] == "decode_attention_tile"
        and "thunderkittens_mha_capture.py" in command["command"]
        and "--paper-baseline-run-id thunderkittens_decode_attention_tile"
        in command["command"]
        and "--benchmark-id llm_serving_decode" in command["command"]
        and "--serving-workload-id vdcores_offline_decode" in command["command"]
        and "--prompt-tokens 128" in command["command"]
        and "--decode-tokens 64" in command["command"]
        and "--shape 4,1,256,64" in command["command"]
        for command in thunderkittens_decode["commands"]
    )
    assert all(
        command.get("raw_artifact", "").startswith(
            "tmp/cuda-backend/paper-baselines/serving-runs"
        )
        for record in records
        for command in record["commands"]
        if command["kind"] != "server"
    )


def test_paper_baseline_pair_probe_uses_remote_fallback_contract():
    script_path = (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_pair_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "paper_baseline_pair_probe",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = module.PairedPaperBaselineProbeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="goal/nvidia-paper-ready",
        output_root=Path("tmp/cuda-backend/paper-baselines/probes"),
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
        refresh_remote=False,
        sync_remote_tree=True,
    )

    sync_command = module.build_remote_sync_command(config)
    assert sync_command[:3] == ["rsync", "-a", "--delete"]
    assert "--exclude=.venv" in sync_command
    assert "--exclude=build" in sync_command
    assert "--exclude=tmp" in sync_command
    assert sync_command[-1] == "h200-box:/remote/pto-cu/"

    source_sync_command = module.build_remote_baseline_source_sync_command(config)
    assert source_sync_command[:3] == ["rsync", "-a", "--delete"]
    assert "--exclude=build/" in source_sync_command
    assert "--exclude=*.egg-info/" in source_sync_command
    assert "--exclude=__pycache__/" in source_sync_command
    assert "--exclude=*.pyc" in source_sync_command
    assert "--exclude=*.cpython-*.so" in source_sync_command
    assert source_sync_command[-2] == "tmp/baselines/"
    assert source_sync_command[-1] == "h200-box:/remote/pto-cu/tmp/baselines/"

    remote_command = module.build_remote_probe_command(config, "abc123")
    assert remote_command[:5] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    assert remote_command[5] == "h200-box"
    shell = remote_command[-1]
    assert "cd /remote/pto-cu" in shell
    assert "CUDA_HOME=" in shell
    assert "PATH=" in shell
    assert "PYTHONPATH=$PWD:$PWD/python" in shell
    assert "paper_baseline_probe.py" in shell
    assert "--output" in shell
    assert "h200-probe.json" in shell
    assert "git fetch" not in shell
    assert "git checkout" not in shell

    default_config = module.PairedPaperBaselineProbeConfig(
        remote="h200-box",
        remote_workdir="/remote/pto-cu",
        branch="goal/nvidia-paper-ready",
        local_python=".venv/bin/python",
        remote_python=".venv/bin/python",
    )
    default_shell = module.build_remote_probe_command(default_config, "abc123")[-1]
    assert "fetch origin goal/nvidia-paper-ready" in default_shell
    assert "git checkout -B goal/nvidia-paper-ready FETCH_HEAD" in default_shell


def test_evaluation_docs_are_split_for_review():
    root_evaluation_docs = sorted(DOC_ROOT.glob("evaluation*.md"))
    assert {path.name for path in root_evaluation_docs} == {
        "evaluation-current.md",
        "evaluation.md",
    }
    for path in root_evaluation_docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 220, f"{path} has {len(lines)} lines"

    history_root = DOC_ROOT / "history"
    assert (history_root / "index.md").is_file()
    assert (history_root / "captures" / "current-head-layered-cross-743709f3.md").is_file()
    assert (history_root / "captures" / "legacy-captures.md").is_file()


def test_benchmark_viewer_has_json_backed_review_data():
    assert (VIEWER_ROOT / "index.html").is_file()
    assert (VIEWER_ROOT / "styles.css").is_file()
    assert (VIEWER_ROOT / "viewer.js").is_file()
    assert (VIEWER_ROOT / "data" / "paper_baselines.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_baseline_runs.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_baseline_probes.json").is_file()
    assert (
        VIEWER_ROOT / "data" / "paper_baseline_environment_plans.json"
    ).is_file()
    assert (
        VIEWER_ROOT / "data" / "paper_baseline_environment_attempts.json"
    ).is_file()
    assert (VIEWER_ROOT / "data" / "paper_baseline_run_readiness.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_baseline_execution_attempts.json").is_file()
    assert (VIEWER_ROOT / "data" / "serving_command_plan.json").is_file()
    assert (VIEWER_ROOT / "data" / "serving_workloads.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_readiness_audit.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_readiness_work_queue.json").is_file()
    assert (VIEWER_ROOT / "data" / "goal_progress.json").is_file()
    assert (VIEWER_ROOT / "data" / "capture_imports.json").is_file()
    viewer_js = (VIEWER_ROOT / "viewer.js").read_text(encoding="utf-8")
    for required in [
        "run.inputs.shape",
        "run.inputs.dtype",
        "run.inputs.repeat_policy",
        "method.category",
        "method.launch_model",
        "paperBaselineRuns",
        "paper_baseline_runs",
        "paperBaselineProbes",
        "paper_baseline_probes",
        "paperBaselineEnvironmentPlans",
        "paper_baseline_environment_plans",
        "Environment Plans",
        "preflight_commands",
        "Preflight",
        "paperBaselineEnvironmentAttempts",
        "paper_baseline_environment_attempts",
        "Environment Attempts",
        "paperBaselineRunReadiness",
        "paper_baseline_run_readiness",
        "Run Readiness",
        "paperBaselineExecutionAttempts",
        "paper_baseline_execution_attempts",
        "Execution Attempts",
        "next_actions",
        "Next Actions",
        "servingCommandPlan",
        "serving_command_plan.json",
        "Serving Command Plan",
        "servingWorkloads",
        "serving_workloads",
        "Serving policies",
        "latest_artifact_root",
        "latest_machine_status",
        "paperEvaluation",
        "paper_evaluation_matrix",
        "paperReadinessAudit",
        "paper_readiness_audit",
        "paperReadinessWorkQueue",
        "paper_readiness_work_queue",
        "Paper Work Queue",
        "work_items_by_source",
        "goalProgress",
        "goal_progress",
        "Goal Progress",
        "acceptance_criteria",
        "paper_grade_results",
        "paper_baseline_run_readiness_statuses",
        "ready_for_paper_claim",
        "result_records",
        "raw_artifact",
        "correctness",
    ]:
        assert required in viewer_js

    benchmarks = json.loads(
        (VIEWER_ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8")
    )
    methods = json.loads(
        (VIEWER_ROOT / "data" / "methods.json").read_text(encoding="utf-8")
    )
    paper_baselines = json.loads(
        (VIEWER_ROOT / "data" / "paper_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    paper_baseline_runs = json.loads(
        (VIEWER_ROOT / "data" / "paper_baseline_runs.json").read_text(
            encoding="utf-8"
        )
    )
    paper_baseline_probes = json.loads(
        (VIEWER_ROOT / "data" / "paper_baseline_probes.json").read_text(
            encoding="utf-8"
        )
    )
    paper_baseline_run_readiness = json.loads(
        (VIEWER_ROOT / "data" / "paper_baseline_run_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    paper_baseline_execution_attempts = json.loads(
        (
            VIEWER_ROOT
            / "data"
            / "paper_baseline_execution_attempts.json"
        ).read_text(encoding="utf-8")
    )
    serving_command_plan = json.loads(
        (VIEWER_ROOT / "data" / "serving_command_plan.json").read_text(
            encoding="utf-8"
        )
    )
    serving_workloads = json.loads(
        (VIEWER_ROOT / "data" / "serving_workloads.json").read_text(
            encoding="utf-8"
        )
    )
    paper_evaluation = json.loads(
        (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    paper_readiness_audit = json.loads(
        (VIEWER_ROOT / "data" / "paper_readiness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    paper_readiness_work_queue = json.loads(
        (VIEWER_ROOT / "data" / "paper_readiness_work_queue.json").read_text(
            encoding="utf-8"
        )
    )
    goal_progress = json.loads(
        (VIEWER_ROOT / "data" / "goal_progress.json").read_text(
            encoding="utf-8"
        )
    )
    capture_imports = json.loads(
        (VIEWER_ROOT / "data" / "capture_imports.json").read_text(
            encoding="utf-8"
        )
    )
    results = json.loads(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8")
    )

    benchmark_ids = {item["id"] for item in benchmarks["benchmarks"]}
    assert "llm_serving_decode" in benchmark_ids
    assert "graph_layered_cross" in benchmark_ids
    assert "tensor_core_tile" in benchmark_ids
    for benchmark in benchmarks["benchmarks"]:
        assert benchmark["description"]
        assert benchmark["math"]
        assert benchmark["code"]
        assert benchmark["run"]["command"]
        assert benchmark["run"]["inputs"]
        assert benchmark["evidence_refs"]

    method_ids = {item["id"] for item in methods["methods"]}
    assert {
        "pto_host_schedule",
        "pto_persistent_device",
        "direct_runtime",
        "direct_driver",
        "direct_driver_graph",
        "cublas_sgemm_graph",
        "triton",
        "cutlass",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= method_ids
    for method in methods["methods"]:
        assert method["category"]
        assert method["launch_model"]

    import_baselines = {
        item["baseline"] for item in capture_imports["capture_imports"]
    }
    assert "direct_runtime" in import_baselines
    assert "direct_driver" in import_baselines
    assert "direct_driver_graph" in import_baselines
    assert "direct_driver_sgemm" in import_baselines
    assert "direct_runtime_sgemm" in import_baselines
    assert "direct_driver_graph_sgemm" in import_baselines
    assert "pto_persistent_dag_graph_tensor_core" in import_baselines

    paper_baseline_ids = {
        item["id"] for item in paper_baselines["paper_baselines"]
    }
    assert {"mpk", "vdcores"} <= paper_baseline_ids
    assert "vllm" in paper_baseline_ids
    assert "sglang" in paper_baseline_ids
    for baseline in paper_baselines["paper_baselines"]:
        assert baseline["status"]
        assert baseline["source"]["upstream_url"]
        assert baseline["paper_role"]
        assert baseline["next_action"]

    by_id = {item["id"]: item for item in paper_baselines["paper_baselines"]}
    for baseline_id in ["mpk", "vdcores", "vllm", "sglang", "thunderkittens"]:
        baseline = by_id[baseline_id]
        assert baseline["status"] == "source_cloned_for_survey"
        assert len(baseline["source"]["commit"]) == 40

    serving_by_id = {
        item["id"]: item for item in serving_workloads["serving_workloads"]
    }
    assert {"mpk_offline_decode", "vdcores_offline_decode"} <= set(serving_by_id)
    assert serving_by_id["mpk_offline_decode"]["model_policy"]["primary_model"] == (
        "Qwen/Qwen3-8B"
    )
    assert (
        serving_by_id["mpk_offline_decode"]["prompt_policy"][
            "target_prompt_tokens"
        ]
        == 64
    )
    assert serving_by_id["mpk_offline_decode"]["decode_policy"]["decode_tokens"] == 1024
    assert serving_by_id["vdcores_offline_decode"]["decode_policy"]["decode_tokens"] == 64
    assert serving_by_id["vdcores_offline_decode"]["status"] == (
        "partial_controlled_results"
    )
    assert serving_by_id["vdcores_offline_decode"]["prompt_policy"][
        "target_prompt_tokens"
    ] == 128
    assert any(
        ref["path"] == "docs/nvidia-backend/benchmark-viewer/data/results.json"
        and "pto_controlled_serving_equivalent" in ref["symbols"]
        for ref in serving_by_id["vdcores_offline_decode"]["evidence_refs"]
    )
    for workload in serving_workloads["serving_workloads"]:
        assert workload["baseline_run_ids"]
        assert workload["required_metrics"]
        assert workload["evidence_refs"]

    run_ids = {item["id"] for item in paper_baseline_runs["paper_baseline_runs"]}
    assert {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
        "thunderkittens_decode_attention_tile",
    } <= run_ids
    run_baselines = {
        item["paper_baseline_id"]
        for item in paper_baseline_runs["paper_baseline_runs"]
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= run_baselines
    for item in paper_baseline_runs["paper_baseline_runs"]:
        assert item["paper_baseline_id"] in paper_baseline_ids
        assert item["paper_evaluation_id"]
        assert item["hardware_targets"]
        assert item["setup_commands"]
        assert item["run_commands"]
        assert item["expected_artifacts"]
        assert item["import_target"]["viewer_file"].endswith("results.json")
        if item["paper_baseline_id"] in {"mpk", "vdcores"}:
            assert isinstance(item.get("model_access"), dict)
            assert isinstance(item["model_access"]["requires_hf_token"], bool)
        if item["paper_evaluation_id"] == "llm_serving_paper_baselines":
            assert item["serving_workload_ids"]
            assert {
                "model_and_prompt_shape",
                "batch_or_concurrency_policy",
            } <= set(item["required_metrics"])
        if item["id"] == "thunderkittens_tile_kernel":
            assert item["status"] == "imported_to_viewer"
            assert any(
                "quick-smoke.json" in path for path in item["expected_artifacts"]
            )
            assert any("capture.json" in path for path in item["expected_artifacts"])
            assert any(
                "thunderkittens_mha_capture.py" in command
                for command in item["run_commands"]
            )
            assert not any(
                path.endswith("correctness.json") or path.endswith("benchmark.json")
                for path in item["expected_artifacts"]
            )
        if item["id"] == "thunderkittens_full_sweep":
            assert item["paper_evaluation_id"] == "tensor_core_tile_baselines"
            assert item["status"] == "imported_to_viewer"
            assert "Selected" in item["title"]
            assert item["serving_workload_ids"] == []
            assert any(
                path.endswith("correctness.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("benchmark.json") for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("upstream-summary.json")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("run-after-build.log")
                for path in item["expected_artifacts"]
            )
            assert any(
                path.endswith("correctness.log")
                for path in item["expected_artifacts"]
            )
            assert any(
                "thunderkittens_full_sweep_capture.py" in command
                for command in item["run_commands"]
            )
        if item["id"] == "thunderkittens_decode_attention_tile":
            assert item["paper_evaluation_id"] == "llm_serving_paper_baselines"
            assert item["status"] == "imported_to_viewer"
            assert item["serving_workload_ids"] == ["vdcores_offline_decode"]
            assert any(
                "thunderkittens_mha_capture.py" in command
                and "--shape <batch>,1,256,64" in command
                and "--prompt-tokens 128" in command
                and "--decode-tokens 64" in command
                for command in item["run_commands"]
            )

    assert paper_readiness_work_queue["summary"]["total_work_items"] == sum(
        len(claim["next_actions"])
        for claim in paper_readiness_audit["claim_audits"]
    )
    assert goal_progress["overall_status"] == "in_progress"
    assert any(
        item["id"] == "paper_grade_results"
        and item["status"] == "in_progress"
        and item["blocking_work_items"]
        == paper_readiness_work_queue["summary"]["total_work_items"]
        for item in goal_progress["acceptance_criteria"]
    )

    probe_baselines = {
        item["paper_baseline_id"]
        for item in paper_baseline_probes["paper_baseline_probes"]
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= probe_baselines
    for item in paper_baseline_probes["paper_baseline_probes"]:
        assert item["latest_artifact_root"].startswith("tmp/")
        probe_root = ROOT / item["latest_artifact_root"]
        assert probe_root.is_dir()
        assert any(path.suffix == ".json" for path in probe_root.iterdir())
        assert (
            item["latest_artifact_root"]
            == "tmp/cuda-backend/paper-baselines/probes/"
            "paired-a100-h200-86ea3913/"
        )
        assert item["checks"]
        assert item["next_action"]
        machine_status = {
            status["gpu"]: status for status in item["latest_machine_status"]
        }
        assert set(machine_status) == {"A100", "H200"}
        for status in machine_status.values():
            assert status["status"] in {"pass", "partial", "fail", "not_captured"}
            artifact = ROOT / status["artifact"]
            assert artifact.is_file()
            assert artifact.suffix == ".json"
            assert isinstance(status["blocking_gaps"], list)
        if item["paper_baseline_id"] == "thunderkittens":
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert {
                "torch",
                "pybind11",
                "numpy",
                "pandas",
                "matplotlib",
                "tqdm",
            } <= probed_modules
        if item["paper_baseline_id"] in {"mpk", "vdcores"}:
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert "transformers" in probed_modules
        if item["paper_baseline_id"] == "mpk":
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert "mirage.mpk.base_dynamic_shard_loader" in imported_modules
        if item["paper_baseline_id"] == "sglang":
            probed_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_module"
            }
            assert {"orjson", "torchvision"} <= probed_modules
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert {
                "sglang.bench_serving",
                "sglang.bench_offline_throughput",
                "sglang.bench_one_batch",
            } <= imported_modules
            assert all(
                check.get("python_no_user_site") is True
                for check in item["checks"]
                if check["kind"] == "python_import"
            )
        if item["paper_baseline_id"] == "vllm":
            imported_modules = {
                check["module"]
                for check in item["checks"]
                if check["kind"] == "python_import"
            }
            assert {
                "vllm",
                "vllm.entrypoints.cli.main",
                "vllm.entrypoints.openai.api_server",
                "vllm.engine.arg_utils",
            } <= imported_modules
            assert all(
                check.get("python_no_user_site") is True
                for check in item["checks"]
                if check["kind"] == "python_import"
            )
    readiness_by_run = {
        item["paper_baseline_run_id"]: item
        for item in paper_baseline_run_readiness["paper_baseline_run_readiness"]
    }
    execution_attempts = paper_baseline_execution_attempts[
        "paper_baseline_execution_attempts"
    ]
    attempts_by_id = {item["id"]: item for item in execution_attempts}
    assert {
        "mpk_qwen3_0p6b_native_token2_h200",
        "mpk_qwen3_0p6b_persistent_profile_token2_h200",
        "mpk_qwen3_0p6b_persistent_noprofile_token2_h200",
        "mpk_qwen3_0p6b_persistent_token8_h200",
        "mpk_qwen3_0p6b_launch_blocking_token2_h200",
        "mpk_qwen3_0p6b_no_cutlass_token2_h200",
        "mpk_qwen3_0p6b_token1_noprofile_h200",
        "mpk_qwen3_0p6b_token1_memcheck_h200",
        "mpk_qwen3_0p6b_snapshot_pointer_patch_token1_h200",
        "mpk_qwen3_0p6b_snapshot_pointer_patch_memcheck_h200",
        "mpk_qwen3_0p6b_snapshot_pointer_patch_predecode_memcheck_h200",
        "mpk_qwen3_0p6b_workload_metadata_sweep_h200",
        "mpk_qwen3_0p6b_bounded_decode_h200",
        "mpk_qwen3_0p6b_bounded_profile_diagnostic_h200",
        "mpk_qwen3_0p6b_profile_noop_diagnostic_h200",
        "mpk_qwen3_0p6b_profile_termination_diagnostic_h200",
        "vdcores_qwen3_1p7b_dry_build_h200",
        "vdcores_qwen3_1p7b_correctness_hf_timeout_h200",
        "vdcores_qwen3_1p7b_selected_runtime_rebuild_h200",
        "vdcores_qwen3_1p7b_correctness_rebuilt_illegal_memory_h200",
        "vdcores_qwen3_1p7b_stage_launch_sweep_h200",
        "vdcores_qwen3_1p7b_final_rms_memcheck_h200",
        "vdcores_qwen3_1p7b_no_prefetch_sweep_h200",
        "vdcores_qwen3_1p7b_minst_provenance_h200",
        "vdcores_qwen3_1p7b_runtime_ldwarp_sm64_h200",
        "vdcores_qwen3_1p7b_pointer_attr_probe_h200",
        "vdcores_qwen3_1p7b_launch_pointer_attr_probe_h200",
        "vdcores_qwen3_1p7b_single_visible_gpu_h200",
        "vdcores_qwen3_1p7b_logits_stage_bisect_h200",
        "vdcores_qwen3_1p7b_logits_schedule_introspection_h200",
        "vdcores_qwen3_1p7b_slot_repeat_source_analysis_h200",
        "vdcores_qwen3_1p7b_slot_lifetime_pc44_pc52_h200",
        "vdcores_qwen3_1p7b_repeat_lane_source_diagnostic_h200",
        "vdcores_qwen3_1p7b_repeat_state_lite_h200",
        "vdcores_qwen3_1p7b_repeat_guard_bench_h200",
        "vdcores_qwen3_1p7b_repeat_guard_correctness_h200",
        "thunderkittens_mha_h100_official_benchmark_h200",
    } <= set(attempts_by_id)
    assert attempts_by_id["mpk_qwen3_0p6b_native_token2_h200"]["status"] == "pass"
    assert (
        attempts_by_id["mpk_qwen3_0p6b_native_token2_h200"]["summary"][
            "generate_length"
        ]
        == 2
    )
    profile_attempt = attempts_by_id[
        "mpk_qwen3_0p6b_persistent_profile_token2_h200"
    ]
    assert profile_attempt["status"] == "failed_after_kernel_launch"
    assert "KeyError: (16, 0)" in profile_attempt["blocker"]
    assert profile_attempt["summary"]["compiled"]
    assert profile_attempt["summary"]["launched"]
    assert profile_attempt["summary"]["total_tasks"] == 7261
    assert "illegal memory access" in attempts_by_id[
        "mpk_qwen3_0p6b_persistent_noprofile_token2_h200"
    ]["blocker"]
    launch_blocking_attempt = attempts_by_id[
        "mpk_qwen3_0p6b_launch_blocking_token2_h200"
    ]
    assert launch_blocking_attempt["status"] == "blocked"
    assert "CUDA_LAUNCH_BLOCKING=1" in launch_blocking_attempt["blocker"]
    assert launch_blocking_attempt["summary"]["terminated"]
    assert launch_blocking_attempt["summary"]["compiled_artifacts_present"]
    no_cutlass_attempt = attempts_by_id["mpk_qwen3_0p6b_no_cutlass_token2_h200"]
    assert no_cutlass_attempt["status"] == "failed_after_kernel_launch"
    assert "no-CUTLASS" in no_cutlass_attempt["blocker"]
    assert "illegal memory access" in no_cutlass_attempt["blocker"]
    assert not no_cutlass_attempt["summary"]["use_cutlass_kernel"]
    assert no_cutlass_attempt["summary"]["launched"]
    one_token_attempt = attempts_by_id["mpk_qwen3_0p6b_token1_noprofile_h200"]
    assert one_token_attempt["status"] == "failed_after_kernel_launch"
    assert one_token_attempt["summary"]["max_new_tokens"] == 1
    assert "illegal memory access" in one_token_attempt["blocker"]
    mpk_memcheck = attempts_by_id["mpk_qwen3_0p6b_token1_memcheck_h200"]
    assert mpk_memcheck["status"] == "failed_after_kernel_launch"
    assert mpk_memcheck["summary"]["tool"] == "compute-sanitizer memcheck"
    assert mpk_memcheck["summary"]["offline_cache_used"]
    assert mpk_memcheck["summary"]["sanitizer_exit_status"] == 1
    assert mpk_memcheck["summary"]["error_summary_count"] == 4
    assert mpk_memcheck["summary"]["total_tasks"] == 7261
    first_mpk_invalid = mpk_memcheck["summary"]["first_invalid_access"]
    assert first_mpk_invalid["function"] == "prepare_next_batch"
    assert first_mpk_invalid["source"] == "persistent_kernel.cuh:273"
    assert first_mpk_invalid["address"] == "0x0"
    assert first_mpk_invalid["device_source"] == "persistent_kernel.cuh:1235"
    assert first_mpk_invalid["kernel_source"] == "persistent_kernel.cuh:1391"
    assert "paged_kv_indices_snapshot" in mpk_memcheck["blocker"]
    patched_mpk = attempts_by_id[
        "mpk_qwen3_0p6b_snapshot_pointer_patch_token1_h200"
    ]
    assert patched_mpk["status"] == "partial"
    assert patched_mpk["summary"]["null_snapshot_write_cleared"]
    assert patched_mpk["summary"]["tokens_saved"]
    assert patched_mpk["summary"]["exit_status"] == 0
    assert "local patch" in patched_mpk["blocker"]
    runtime_patch = (
        "docs/nvidia-backend/baseline-patches/"
        "mpk-snapshot-pointer-runtime-config.patch"
    )
    assert runtime_patch in patched_mpk["reproducibility_patches"]
    assert "paged_kv_indices_snapshot" in (ROOT / runtime_patch).read_text(
        encoding="utf-8"
    )
    patched_mpk_memcheck = attempts_by_id[
        "mpk_qwen3_0p6b_snapshot_pointer_patch_memcheck_h200"
    ]
    assert patched_mpk_memcheck["status"] == "partial"
    assert patched_mpk_memcheck["summary"]["tool"] == "compute-sanitizer memcheck"
    assert patched_mpk_memcheck["summary"]["null_snapshot_write_cleared"]
    assert patched_mpk_memcheck["summary"]["invalid_global_access_count"] == 0
    assert patched_mpk_memcheck["summary"]["error_summary_count"] == 1
    assert (
        patched_mpk_memcheck["summary"]["application_error"]["type"]
        == "OverflowError"
    )
    assert "snapshot pointer patch" in patched_mpk_memcheck["blocker"]
    assert runtime_patch in patched_mpk_memcheck["reproducibility_patches"]
    patched_mpk_predecode = attempts_by_id[
        "mpk_qwen3_0p6b_snapshot_pointer_patch_predecode_memcheck_h200"
    ]
    assert patched_mpk_predecode["status"] == "partial"
    assert patched_mpk_predecode["summary"]["tool"] == "compute-sanitizer memcheck"
    assert patched_mpk_predecode["summary"]["null_snapshot_write_cleared"]
    assert patched_mpk_predecode["summary"]["invalid_global_access_count"] == 0
    assert patched_mpk_predecode["summary"]["predecode_dump_written"]
    assert patched_mpk_predecode["summary"]["first_generated_token_id"] == -1
    assert patched_mpk_predecode["summary"]["generated_length"] == 1
    assert (
        patched_mpk_predecode["summary"]["application_error"]["type"]
        == "OverflowError"
    )
    assert "generated token -1" in patched_mpk_predecode["blocker"]
    debug_patch = (
        "docs/nvidia-backend/baseline-patches/"
        "mpk-predecode-token-dump.patch"
    )
    assert patched_mpk_predecode["reproducibility_patches"] == [
        runtime_patch,
        debug_patch,
    ]
    assert "tokens.json.predecode.json" in " ".join(
        patched_mpk_predecode["artifacts"]
    )
    assert "generated_token_ids" in (ROOT / debug_patch).read_text(
        encoding="utf-8"
    )
    mpk_workload = attempts_by_id[
        "mpk_qwen3_0p6b_workload_metadata_sweep_h200"
    ]
    assert mpk_workload["status"] == "partial"
    assert mpk_workload["summary"]["mode"] == (
        "mpk_qwen3_0p6b_workload_metadata_sweep"
    )
    assert mpk_workload["summary"]["requested_max_new_tokens"] == [1, 2]
    assert mpk_workload["summary"]["observed_generate_lengths"] == [89, 89]
    assert mpk_workload["summary"]["observed_predecode_steps"] == [127, 127]
    assert not mpk_workload["summary"]["max_new_tokens_honored"]
    assert mpk_workload["summary"]["persistent_loop_runs_to_max_seq_length"]
    assert mpk_workload["summary"]["all_runs_exit_zero"]
    assert mpk_workload["summary"]["total_tasks"] == [7261, 7261]
    assert mpk_workload["summary"]["total_events"] == [1870, 1870]
    assert "matching workload" in mpk_workload["blocker"]
    assert "workload-summary.json" in " ".join(mpk_workload["artifacts"])
    mpk_bounded = attempts_by_id["mpk_qwen3_0p6b_bounded_decode_h200"]
    assert mpk_bounded["status"] == "partial"
    assert mpk_bounded["summary"]["mode"] == "mpk_qwen3_0p6b_bounded_decode"
    assert mpk_bounded["summary"]["requested_max_new_tokens"] == [1, 2]
    assert mpk_bounded["summary"]["bounded_max_seq_lengths"] == [40, 41]
    assert mpk_bounded["summary"]["observed_prompt_lengths"] == [39, 39]
    assert mpk_bounded["summary"]["observed_generate_lengths"] == [1, 2]
    assert mpk_bounded["summary"]["observed_predecode_steps"] == [39, 40]
    assert mpk_bounded["summary"]["requested_decode_lengths_honored"]
    assert mpk_bounded["summary"]["max_seq_length_used_as_decode_bound"]
    assert mpk_bounded["summary"]["all_runs_exit_zero"]
    assert mpk_bounded["summary"]["total_tasks"] == [7261, 7261]
    assert mpk_bounded["summary"]["total_events"] == [1870, 1870]
    assert mpk_bounded["summary"]["scheduler_trace_available"]
    assert not mpk_bounded["summary"]["resource_policy_measured"]
    assert "bounded-decode-summary.json" in " ".join(mpk_bounded["artifacts"])
    assert "scheduler/resource/latency" in mpk_bounded["blocker"]
    mpk_profile = attempts_by_id[
        "mpk_qwen3_0p6b_bounded_profile_diagnostic_h200"
    ]
    profile_patch = (
        "docs/nvidia-backend/baseline-patches/"
        "mpk-profiler-trace-export-diagnostic.patch"
    )
    assert mpk_profile["status"] == "partial"
    assert mpk_profile["summary"]["mode"] == (
        "mpk_qwen3_0p6b_bounded_profile_diagnostic"
    )
    assert mpk_profile["summary"]["profile_export_without_patch_status"] == 1
    assert mpk_profile["summary"]["profile_export_key_error"] == "KeyError: (16, 0)"
    assert mpk_profile["summary"]["profile_export_patch_status"] == 0
    assert mpk_profile["summary"]["large_buffer_status"] == 0
    assert mpk_profile["summary"]["perfetto_trace_exported"]
    assert mpk_profile["summary"]["perfetto_trace_bytes"] > 400000
    assert mpk_profile["summary"]["profiled_run_generated_length"] == 0
    assert mpk_profile["summary"]["profiled_predecode_step"] == 1
    assert not mpk_profile["summary"]["profiling_preserves_correctness"]
    assert "mpk_bounded_decode.perfetto-trace" in " ".join(
        mpk_profile["artifacts"]
    )
    assert profile_patch in mpk_profile["reproducibility_patches"]
    profile_patch_text = (ROOT / profile_patch).read_text(encoding="utf-8")
    assert "get_tid" in profile_patch_text
    assert "torch.cuda.synchronize" in profile_patch_text
    assert "30000 * 256" in profile_patch_text
    assert "profiling and correctness do not yet coexist" in mpk_profile["blocker"]
    mpk_noop = attempts_by_id["mpk_qwen3_0p6b_profile_noop_diagnostic_h200"]
    all_noop_patch = (
        "docs/nvidia-backend/baseline-patches/"
        "mpk-profiler-all-noop-diagnostic.patch"
    )
    assert mpk_noop["status"] == "partial"
    assert mpk_noop["summary"]["mode"] == (
        "mpk_qwen3_0p6b_profile_noop_diagnostic"
    )
    assert mpk_noop["summary"]["profile_event_noop_status"] == 0
    assert mpk_noop["summary"]["profile_all_macros_noop_status"] == 0
    assert mpk_noop["summary"]["profile_event_noop_generated_length"] == 0
    assert mpk_noop["summary"]["profile_all_macros_noop_generated_length"] == 0
    assert mpk_noop["summary"]["profile_event_noop_predecode_step"] == 1
    assert mpk_noop["summary"]["profile_all_macros_noop_predecode_step"] == 1
    assert not mpk_noop["summary"]["event_writes_required_for_corruption"]
    assert mpk_noop["summary"]["all_profiler_macros_disabled"]
    assert mpk_noop["summary"]["profile_compile_path_still_corrupts_token_progress"]
    assert not mpk_noop["summary"]["profiling_preserves_correctness"]
    assert "profile_all_macros_noop" in " ".join(mpk_noop["artifacts"])
    assert all_noop_patch in mpk_noop["reproducibility_patches"]
    all_noop_patch_text = (ROOT / all_noop_patch).read_text(encoding="utf-8")
    assert "PROFILER_CLOSURE_PARAMS_DECL" in all_noop_patch_text
    assert "PROFILER_INIT" in all_noop_patch_text
    assert "profile compile path still corrupts token progress" in mpk_noop[
        "blocker"
    ]
    mpk_termination = attempts_by_id[
        "mpk_qwen3_0p6b_profile_termination_diagnostic_h200"
    ]
    termination_patch = (
        "docs/nvidia-backend/baseline-patches/"
        "mpk-profiler-termination-diagnostic.patch"
    )
    assert mpk_termination["status"] == "pass"
    assert mpk_termination["summary"]["mode"] == (
        "mpk_qwen3_0p6b_profile_termination_diagnostic"
    )
    assert mpk_termination["summary"]["profile_compile_flag_enabled"]
    assert mpk_termination["summary"]["early_done_override_removed"]
    assert mpk_termination["summary"]["all_noop_status"] == 0
    assert mpk_termination["summary"]["real_profiler_status"] == 0
    assert mpk_termination["summary"]["all_noop_generated_length"] == 2
    assert mpk_termination["summary"]["real_profiler_generated_length"] == 2
    assert mpk_termination["summary"]["all_noop_predecode_step"] == 40
    assert mpk_termination["summary"]["real_profiler_predecode_step"] == 40
    assert mpk_termination["summary"]["all_noop_trace_bytes"] == 89
    assert mpk_termination["summary"]["real_profiler_trace_bytes"] > 16000000
    assert mpk_termination["summary"]["profiling_preserves_correctness"]
    assert mpk_termination["summary"]["scheduler_trace_available"]
    assert not mpk_termination["summary"]["resource_policy_measured"]
    assert "profile_termination_normalized_real_profiler" in " ".join(
        mpk_termination["artifacts"]
    )
    assert termination_patch in mpk_termination["reproducibility_patches"]
    termination_patch_text = (ROOT / termination_patch).read_text(
        encoding="utf-8"
    )
    assert "MPK_ENABLE_PROFILING" in termination_patch_text
    assert "if (true)" in termination_patch_text
    vdcores_attempt = attempts_by_id["vdcores_qwen3_1p7b_dry_build_h200"]
    assert vdcores_attempt["status"] == "partial"
    assert vdcores_attempt["summary"]["layers"] == 28
    assert vdcores_attempt["summary"]["compute_operators"] == 9
    assert not vdcores_attempt["summary"]["requires_hf_token"]
    vdcores_correctness_attempt = attempts_by_id[
        "vdcores_qwen3_1p7b_correctness_hf_timeout_h200"
    ]
    assert vdcores_correctness_attempt["status"] == "blocked_before_model_load"
    assert vdcores_correctness_attempt["summary"]["hf_fetch_timeout"]
    assert vdcores_correctness_attempt["summary"]["dae_runtime_present"]
    assert not vdcores_correctness_attempt["summary"]["model_loaded"]
    assert not vdcores_correctness_attempt["summary"]["launched"]
    assert not vdcores_correctness_attempt["summary"]["resource_policy_measured"]
    assert "config.json timed out" in vdcores_correctness_attempt["blocker"]
    vdcores_rebuild_attempt = attempts_by_id[
        "vdcores_qwen3_1p7b_selected_runtime_rebuild_h200"
    ]
    assert vdcores_rebuild_attempt["status"] == "pass"
    assert vdcores_rebuild_attempt["summary"]["selected_compute_ops"] == 9
    assert vdcores_rebuild_attempt["summary"]["all_required_compute_ops_supported"]
    assert vdcores_rebuild_attempt["summary"]["extension_installed"]
    vdcores_rebuilt_correctness_attempt = attempts_by_id[
        "vdcores_qwen3_1p7b_correctness_rebuilt_illegal_memory_h200"
    ]
    assert vdcores_rebuilt_correctness_attempt["status"] == (
        "failed_after_kernel_launch"
    )
    assert vdcores_rebuilt_correctness_attempt["summary"]["model_loaded"]
    assert vdcores_rebuilt_correctness_attempt["summary"][
        "runtime_rebuilt_with_required_ops"
    ]
    assert vdcores_rebuilt_correctness_attempt["summary"]["launched"]
    assert not vdcores_rebuilt_correctness_attempt["summary"][
        "correctness_completed"
    ]
    assert "illegal memory access" in vdcores_rebuilt_correctness_attempt["blocker"]
    vdcores_stage_sweep = attempts_by_id[
        "vdcores_qwen3_1p7b_stage_launch_sweep_h200"
    ]
    assert vdcores_stage_sweep["status"] == "failed_after_kernel_launch"
    assert vdcores_stage_sweep["summary"]["earliest_failed_stage"] == "final_rms"
    assert set(vdcores_stage_sweep["summary"]["stages"].values()) == {
        "failed_after_kernel_launch"
    }
    assert "first launched VDCores stage" in vdcores_stage_sweep["blocker"]
    vdcores_memcheck = attempts_by_id[
        "vdcores_qwen3_1p7b_final_rms_memcheck_h200"
    ]
    assert vdcores_memcheck["status"] == "failed_after_kernel_launch"
    assert vdcores_memcheck["summary"]["tool"] == "compute-sanitizer memcheck"
    assert vdcores_memcheck["summary"]["debug_stop_after"] == "final_rms"
    assert vdcores_memcheck["summary"]["sanitizer_exit_status"] == 99
    assert vdcores_memcheck["summary"]["error_summary_count"] == 130
    assert vdcores_memcheck["summary"]["invalid_read_size_bytes"] == 4096
    first_invalid = vdcores_memcheck["summary"]["first_invalid_read"]
    assert first_invalid["device_source"] == "ldwarp.cuh:52"
    assert first_invalid["kernel_source"] == "dae2.cuh:167"
    assert "cp_async_bulk" in first_invalid["function"]
    assert (
        vdcores_memcheck["summary"]["warp_illegal_address"]["device_source"]
        == "ldwarp.cuh:113"
    )
    assert "invalid 4096-byte global reads" in vdcores_memcheck["blocker"]
    vdcores_no_prefetch = attempts_by_id[
        "vdcores_qwen3_1p7b_no_prefetch_sweep_h200"
    ]
    assert vdcores_no_prefetch["status"] == "failed_after_kernel_launch"
    assert vdcores_no_prefetch["summary"]["mode"] == (
        "vdcores_qwen_no_prefetch_sweep"
    )
    assert vdcores_no_prefetch["summary"]["debug_stop_after"] == "final_rms"
    assert set(vdcores_no_prefetch["summary"]["failed_variants"]) == {
        "all",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "gate_low",
        "gate_high",
        "up_low",
        "up_high",
        "down_proj",
    }
    assert vdcores_no_prefetch["summary"]["all_variants_loaded_model"]
    assert vdcores_no_prefetch["summary"][
        "all_variants_failed_after_kernel_launch"
    ]
    all_memcheck = vdcores_no_prefetch["summary"]["all_no_prefetch_memcheck"]
    assert all_memcheck["tool"] == "compute-sanitizer memcheck"
    assert all_memcheck["error_summary_count"] == 18
    assert all_memcheck["invalid_4096_byte_global_read"]
    assert all_memcheck["cp_async_bulk_present"]
    assert (
        vdcores_no_prefetch["summary"]["previous_memcheck_error_summary_count"]
        == 130
    )
    assert "per-stage async prefetch routing" in vdcores_no_prefetch["blocker"]
    assert (
        "MInst load address"
        in vdcores_no_prefetch["summary"]["next_debug_target"]
    )
    vdcores_minst = attempts_by_id[
        "vdcores_qwen3_1p7b_minst_provenance_h200"
    ]
    assert vdcores_minst["status"] == "partial"
    assert vdcores_minst["summary"]["mode"] == "vdcores_qwen_minst_provenance"
    assert vdcores_minst["summary"]["token_id"] == 52
    assert vdcores_minst["summary"]["token_embedding_row_bytes"] == 4096
    assert vdcores_minst["summary"]["sampled_sms"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
    ]
    assert vdcores_minst["summary"]["direct_1d_loads_without_owner"] == 0
    assert vdcores_minst["summary"]["sm64_to_sm71_embedding_load_pc"] == 1
    assert vdcores_minst["summary"]["sm0_rms_weight_load_pc"] == 0
    assert vdcores_minst["summary"]["sm0_hidden_load_pcs"] == [23, 54]
    assert "live tensor ranges before launch" in vdcores_minst["blocker"]
    vdcores_ldwarp = attempts_by_id[
        "vdcores_qwen3_1p7b_runtime_ldwarp_sm64_h200"
    ]
    assert vdcores_ldwarp["status"] == "failed_after_kernel_launch"
    assert vdcores_ldwarp["summary"]["mode"] == (
        "vdcores_qwen_runtime_ldwarp_sm64"
    )
    assert vdcores_ldwarp["summary"]["debug_sm"] == 64
    assert vdcores_ldwarp["summary"]["debug_stop_after"] == "final_rms"
    assert vdcores_ldwarp["summary"]["launch_executed"]
    assert vdcores_ldwarp["summary"]["debug_build_status"] == 0
    assert vdcores_ldwarp["summary"]["sanitizer_exit_status"] == 1
    assert vdcores_ldwarp["summary"]["error_summary_count"] == 18
    assert (
        vdcores_ldwarp["summary"]["invalid_4096_byte_global_read_count"]
        == 16
    )
    first_load = vdcores_ldwarp["summary"]["first_ldwarp_load"]
    first_invalid = vdcores_ldwarp["summary"]["first_invalid_read"]
    assert first_load["address"] == first_invalid["address"]
    assert first_load["size_bytes"] == 4096
    assert first_invalid["size_bytes"] == 4096
    assert first_invalid["device_source"] == "ldwarp.cuh:56"
    assert first_invalid["kernel_source"] == "dae2.cuh:167"
    assert vdcores_ldwarp["summary"][
        "runtime_ldwarp_address_matches_memcheck_first_invalid"
    ]
    assert "runtime load warp consumes the same direct 1D address" in (
        vdcores_ldwarp["blocker"]
    )
    vdcores_pointer_attr = attempts_by_id[
        "vdcores_qwen3_1p7b_pointer_attr_probe_h200"
    ]
    assert vdcores_pointer_attr["status"] == "partial"
    assert vdcores_pointer_attr["summary"]["mode"] == (
        "vdcores_qwen_pointer_attr_probe"
    )
    assert not vdcores_pointer_attr["summary"]["launch_executed"]
    assert vdcores_pointer_attr["summary"]["direct_record_count"] == 80
    assert vdcores_pointer_attr["summary"]["direct_load_record_count"] == 40
    assert (
        vdcores_pointer_attr["summary"]["recognized_direct_record_count"]
        == 0
    )
    assert (
        vdcores_pointer_attr["summary"]["unrecognized_direct_record_count"]
        == 80
    )
    assert (
        vdcores_pointer_attr["summary"][
            "device_to_host_copy_ok_record_count"
        ]
        == 80
    )
    assert (
        vdcores_pointer_attr["summary"][
            "direct_load_device_to_host_copy_ok_count"
        ]
        == 40
    )
    sm64_first_direct = vdcores_pointer_attr["summary"][
        "sm64_first_direct_load"
    ]
    assert sm64_first_direct["pc"] == 1
    assert sm64_first_direct["size_bytes"] == 4096
    assert sm64_first_direct["runtime_accumulator"] == 212992
    assert sm64_first_direct["range_status"] == 201
    assert not sm64_first_direct["range_recognized"]
    assert sm64_first_direct["copy_status"] == 0
    assert sm64_first_direct["copy_ok"]
    assert sm64_first_direct["copy_bytes_requested"] == 4096
    assert "cudaMemcpy before launch" in vdcores_pointer_attr["blocker"]
    vdcores_launch_pointer = attempts_by_id[
        "vdcores_qwen3_1p7b_launch_pointer_attr_probe_h200"
    ]
    assert vdcores_launch_pointer["status"] == "failed_after_kernel_launch"
    assert vdcores_launch_pointer["summary"]["mode"] == (
        "vdcores_qwen_launch_pointer_attr_probe"
    )
    assert vdcores_launch_pointer["summary"]["launch_executed"]
    assert vdcores_launch_pointer["summary"]["current_device_before_launch"] == 0
    assert vdcores_launch_pointer["summary"]["direct_record_count"] == 80
    assert vdcores_launch_pointer["summary"][
        "pointer_attribute_success_count"
    ] == 80
    assert vdcores_launch_pointer["summary"][
        "device_to_host_copy_ok_count"
    ] == 80
    assert vdcores_launch_pointer["summary"]["direct_load_pointer_devices"] == [
        0,
        7,
    ]
    assert vdcores_launch_pointer["summary"]["direct_store_pointer_devices"] == [
        0
    ]
    assert (
        vdcores_launch_pointer["summary"]["sm64_first_direct_load_device"]
        == 7
    )
    assert "illegal memory access" in vdcores_launch_pointer["summary"][
        "launch_exception"
    ]
    assert "device-placement mismatch" in vdcores_launch_pointer["blocker"]
    vdcores_single_visible = attempts_by_id[
        "vdcores_qwen3_1p7b_single_visible_gpu_h200"
    ]
    assert vdcores_single_visible["status"] == "partial"
    assert vdcores_single_visible["summary"]["mode"] == (
        "vdcores_qwen_single_visible_gpu"
    )
    assert vdcores_single_visible["summary"]["cuda_visible_devices"] == "7"
    assert vdcores_single_visible["summary"]["final_rms_launch_status"] == 0
    assert vdcores_single_visible["summary"][
        "final_rms_completed_without_python_exception"
    ]
    assert vdcores_single_visible["summary"]["full_one_layer_launch_status"] == 1
    assert "illegal instruction" in vdcores_single_visible["summary"][
        "full_one_layer_error"
    ]
    assert "one-layer final_rms failure" in " ".join(
        vdcores_single_visible["summary"]["rules_out"]
    )
    vdcores_logits_bisect = attempts_by_id[
        "vdcores_qwen3_1p7b_logits_stage_bisect_h200"
    ]
    assert vdcores_logits_bisect["status"] == "failed_after_kernel_launch"
    assert vdcores_logits_bisect["summary"]["mode"] == (
        "vdcores_qwen_logits_stage_bisect"
    )
    assert vdcores_logits_bisect["summary"]["cuda_visible_devices"] == "7"
    assert vdcores_logits_bisect["summary"]["first_failing_stage"] == "logits"
    assert vdcores_logits_bisect["summary"]["stage_statuses"] == {
        "logits": 1,
        "argmax": 1,
        "restore": 1,
        "full": 1,
    }
    assert vdcores_logits_bisect["summary"]["default_logits_split_m"] == 6
    assert vdcores_logits_bisect["summary"]["default_logits_reaches_kernel"]
    assert vdcores_logits_bisect["summary"][
        "default_logits_has_illegal_instruction"
    ]
    assert vdcores_logits_bisect["summary"][
        "default_logits_has_invalid_slot_or_coord_signal"
    ]
    assert vdcores_logits_bisect["summary"]["split_statuses"] == {
        "1": 1,
        "2": 1,
        "3": 1,
        "6": 1,
    }
    assert vdcores_logits_bisect["summary"]["split_1_2_3_fail_before_launch"]
    assert "auto-folding placement assertion" in vdcores_logits_bisect[
        "summary"
    ]["split_1_2_3_failure"]
    assert "logits projection scheduling" in vdcores_logits_bisect["blocker"]
    assert "argmax-only or restore-only" in " ".join(
        vdcores_logits_bisect["summary"]["rules_out"]
    )
    vdcores_logits_schedule = attempts_by_id[
        "vdcores_qwen3_1p7b_logits_schedule_introspection_h200"
    ]
    assert vdcores_logits_schedule["status"] == "blocked"
    assert vdcores_logits_schedule["summary"]["mode"] == (
        "vdcores_qwen_logits_schedule_introspection"
    )
    assert not vdcores_logits_schedule["summary"]["launch_executed"]
    assert vdcores_logits_schedule["summary"]["logits_epoch"] == 3
    assert vdcores_logits_schedule["summary"]["logits_slice"] == 50688
    assert vdcores_logits_schedule["summary"]["vocab_size"] == 151936
    assert vdcores_logits_schedule["summary"]["descriptor_map"] == {
        "32": "logits GEMV loadA matLogitsW[epoch]",
        "33": "logits GEMV loadB matRMSHidden",
        "34": "logits GEMV storeC matLogits[epoch]",
    }
    assert vdcores_logits_schedule["summary"]["sm64_first_logits_pc"] == 38
    assert vdcores_logits_schedule["summary"][
        "sm64_desc32_load_count_in_window"
    ] == 12
    assert vdcores_logits_schedule["summary"][
        "sm64_desc33_load_count_in_window"
    ] == 3
    assert vdcores_logits_schedule["summary"][
        "sm64_desc34_store_count_in_window"
    ] == 2
    assert "RepeatM loop handling" in vdcores_logits_schedule["blocker"]
    vdcores_slot_repeat = attempts_by_id[
        "vdcores_qwen3_1p7b_slot_repeat_source_analysis_h200"
    ]
    assert vdcores_slot_repeat["status"] == "blocked"
    assert vdcores_slot_repeat["summary"]["mode"] == (
        "vdcores_qwen_slot_repeat_source_analysis"
    )
    assert vdcores_slot_repeat["summary"]["source_analysis_only"]
    assert not vdcores_slot_repeat["summary"]["launch_executed"]
    assert vdcores_slot_repeat["summary"]["regular_slots"] == 24
    assert vdcores_slot_repeat["summary"]["slot_metadata_hazard_observed"]
    assert vdcores_slot_repeat["summary"]["pc48_desc34_store_slot"] == 0
    assert vdcores_slot_repeat["summary"]["pc50_desc33_load_reuses_slot"] == 0
    assert vdcores_slot_repeat["summary"][
        "pc50_overwrites_st_insts_slot0_before_stu"
    ]
    assert (
        vdcores_slot_repeat["summary"]["stu_unknown_writeback_opcode"]
        == "opcode=0301 op=12"
    )
    assert vdcores_slot_repeat["summary"]["repeat_coord_hazard_observed"]
    assert vdcores_slot_repeat["summary"]["pc50_repeat_updated_address"] == (
        "0x7fffff"
    )
    assert vdcores_slot_repeat["summary"]["pc50_desc33_invalid_cords"] == [
        65535,
        127,
        0,
    ]
    assert vdcores_slot_repeat["summary"][
        "expected_pc50_desc33_cords_from_build_only_schedule"
    ] == [0, 0, 0, 0]
    assert "slot metadata reuse" in vdcores_slot_repeat["blocker"]
    assert "allocwarp gpr lanes" in vdcores_slot_repeat["summary"][
        "next_debug_target"
    ]
    vdcores_slot_lifetime = attempts_by_id[
        "vdcores_qwen3_1p7b_slot_lifetime_pc44_pc52_h200"
    ]
    assert vdcores_slot_lifetime["status"] == "blocked"
    assert vdcores_slot_lifetime["summary"]["mode"] == (
        "vdcores_qwen_slot_lifetime_pc44_pc52_diagnostic"
    )
    assert vdcores_slot_lifetime["summary"]["launch_executed"]
    assert vdcores_slot_lifetime["summary"]["rebuild_status"] == 0
    assert vdcores_slot_lifetime["summary"]["launch_status"] == 1
    assert vdcores_slot_lifetime["summary"]["pc48_desc34_store_slot"] == 0
    assert vdcores_slot_lifetime["summary"]["pc48_wait_flags_before"] == (
        "0x00fffffe"
    )
    assert vdcores_slot_lifetime["summary"]["pc48_wait_flags_after"] == (
        "0x00ffffff"
    )
    assert vdcores_slot_lifetime["summary"]["stu_copied_pc48_opcode"] == "0443"
    assert not vdcores_slot_lifetime["summary"][
        "unknown_writeback_opcode_seen_after_wait"
    ]
    assert vdcores_slot_lifetime["summary"]["pc49_repeat_packed_delta"] == (
        "0x1000000000"
    )
    assert vdcores_slot_lifetime["summary"]["pc50_addr_accum"] == "0x7fffff"
    assert vdcores_slot_lifetime["summary"]["pc50_desc33_invalid_cords"] == [
        65535,
        127,
        0,
        0,
    ]
    assert vdcores_slot_lifetime["summary"]["slot_metadata_hazard_reduced"]
    assert vdcores_slot_lifetime["summary"]["repeat_coord_hazard_remains"]
    assert "RepeatM lane source" in vdcores_slot_lifetime["summary"][
        "next_debug_target"
    ]
    assert "not sufficient" in vdcores_slot_lifetime["blocker"]
    vdcores_repeat_lane = attempts_by_id[
        "vdcores_qwen3_1p7b_repeat_lane_source_diagnostic_h200"
    ]
    assert vdcores_repeat_lane["status"] == "blocked"
    assert vdcores_repeat_lane["summary"]["mode"] == (
        "vdcores_qwen_repeat_lane_source_diagnostic"
    )
    assert vdcores_repeat_lane["summary"]["launch_executed"]
    assert vdcores_repeat_lane["summary"]["rebuild_status"] == 0
    assert vdcores_repeat_lane["summary"]["launch_status"] == 1
    assert vdcores_repeat_lane["summary"]["split_u64_shuffle_variant_enabled"]
    assert vdcores_repeat_lane["summary"]["pc50_loop_counter_2_source_lane"] == 0
    assert vdcores_repeat_lane["summary"]["pc50_loop_counter_2_split_lane0"] == (
        "0x0"
    )
    assert vdcores_repeat_lane["summary"][
        "pc50_loop_counter_2_split_lanes1_to_7"
    ] == "0x7fffff"
    assert vdcores_repeat_lane["summary"]["pc50_desc33_coords_after_split_variant"] == [
        0,
        0,
        0,
    ]
    assert vdcores_repeat_lane["summary"]["pc52_loop_counter_2_source_lane"] == 2
    assert vdcores_repeat_lane["summary"]["pc52_loop_counter_2_split_lane0"] == (
        "0x1fffff00000000"
    )
    assert vdcores_repeat_lane["summary"][
        "pc52_desc32_unexpected_coords_after_split_variant"
    ] == [0, 12544, 3]
    assert vdcores_repeat_lane["summary"]["final_error"] == "illegal memory access"
    assert vdcores_repeat_lane["summary"]["split_shuffle_not_valid_fix"]
    assert "simple uint64 shuffle lowering bug" in vdcores_repeat_lane["blocker"]
    assert "active masks" in vdcores_repeat_lane["summary"]["next_debug_target"]
    vdcores_repeat_state = attempts_by_id[
        "vdcores_qwen3_1p7b_repeat_state_lite_h200"
    ]
    assert vdcores_repeat_state["status"] == "blocked"
    assert vdcores_repeat_state["summary"]["mode"] == (
        "vdcores_qwen_repeat_state_lite_diagnostic"
    )
    assert vdcores_repeat_state["summary"]["launch_executed"]
    assert vdcores_repeat_state["summary"]["rebuild_status"] == 0
    assert vdcores_repeat_state["summary"]["launch_status"] == 1
    assert vdcores_repeat_state["summary"]["pc48_wait_flags_before"] == (
        "0x00fffffe"
    )
    assert vdcores_repeat_state["summary"]["pc48_wait_flags_after"] == (
        "0x00ffffff"
    )
    assert vdcores_repeat_state["summary"]["stu_copied_pc48_opcode"] == "0443"
    assert not vdcores_repeat_state["summary"][
        "unknown_writeback_opcode_seen_after_wait"
    ]
    assert vdcores_repeat_state["summary"]["pc49_repeat_reg_start"] == 0
    assert vdcores_repeat_state["summary"]["pc49_repeat_reg_end"] == 5
    assert vdcores_repeat_state["summary"]["pc49_repeat_size"] == 2
    assert vdcores_repeat_state["summary"]["pc49_repeat_address"] == (
        "0x1000000000"
    )
    assert vdcores_repeat_state["summary"]["pc49_after_lane0_gpr1"] == "0x0"
    assert vdcores_repeat_state["summary"]["pc50_source_lane"] == 0
    assert vdcores_repeat_state["summary"]["pc50_lane0_gpr1_before_shuffle"] == (
        "0x0"
    )
    assert vdcores_repeat_state["summary"]["pc50_addr_accum"] == "0x7fffff"
    assert vdcores_repeat_state["summary"]["pc50_desc33_invalid_cords"] == [
        65535,
        127,
        0,
    ]
    assert vdcores_repeat_state["summary"][
        "repeat_decode_matches_expected_encoding"
    ]
    assert vdcores_repeat_state["summary"][
        "addr_accum_mismatch_after_repeat_decode"
    ]
    assert "incorrect pc49 RepeatM field encoding" in " ".join(
        vdcores_repeat_state["summary"]["rules_out"]
    )
    assert "native 64-bit shuffle/register transport" in (
        vdcores_repeat_state["summary"]["next_debug_target"]
    )
    vdcores_guard_bench = attempts_by_id[
        "vdcores_qwen3_1p7b_repeat_guard_bench_h200"
    ]
    assert vdcores_guard_bench["status"] == "pass"
    assert vdcores_guard_bench["summary"]["mode"] == (
        "vdcores_qwen_repeat_guard_full_benchmark"
    )
    assert vdcores_guard_bench["summary"]["full_launch_status"] == 0
    assert vdcores_guard_bench["summary"]["bench_status"] == 0
    assert vdcores_guard_bench["summary"]["bench_iterations"] == 5
    assert vdcores_guard_bench["summary"]["num_sms"] == 132
    assert vdcores_guard_bench["summary"]["median_execution_time_ns"] == 1778528
    assert vdcores_guard_bench["summary"]["average_execution_time_ns"] == 1779008
    assert vdcores_guard_bench["summary"]["resource_policy_measured"]
    assert not vdcores_guard_bench["summary"]["correctness_completed"]
    assert not vdcores_guard_bench["summary"]["scheduler_overhead_measured"]
    assert "unguarded addr_accum shuffle" in (
        vdcores_guard_bench["summary"]["root_cause_supported"]
    )
    vdcores_guard_correctness = attempts_by_id[
        "vdcores_qwen3_1p7b_repeat_guard_correctness_h200"
    ]
    assert vdcores_guard_correctness["status"] == "pass"
    assert vdcores_guard_correctness["summary"]["mode"] == (
        "vdcores_qwen_repeat_guard_correctness"
    )
    assert vdcores_guard_correctness["summary"]["correctness_completed"]
    assert vdcores_guard_correctness["summary"]["correctness_passed"]
    assert vdcores_guard_correctness["summary"]["correctness_check_count"] == 17
    assert vdcores_guard_correctness["summary"]["final_token"] == "ref=25, dae=25"
    assert not vdcores_guard_correctness["summary"]["scheduler_overhead_measured"]
    assert not vdcores_guard_correctness["summary"]["queue_pressure_measured"]
    assert "VDCores single-token correctness" in " ".join(
        vdcores_guard_correctness["summary"]["rules_out"]
    )
    vdcores_queue_scheduler = attempts_by_id[
        "vdcores_qwen3_1p7b_queue_scheduler_h200"
    ]
    assert vdcores_queue_scheduler["status"] == "pass"
    assert vdcores_queue_scheduler["summary"]["mode"] == (
        "vdcores_qwen_queue_scheduler_diagnostic"
    )
    assert vdcores_queue_scheduler["summary"]["bench_status"] == 0
    assert vdcores_queue_scheduler["summary"]["correctness_status_code"] == 0
    assert vdcores_queue_scheduler["summary"]["scheduler_overhead_measured"]
    assert vdcores_queue_scheduler["summary"]["queue_pressure_measured"]
    assert vdcores_queue_scheduler["summary"]["max_live_slot_occupancy"] == 24
    assert vdcores_queue_scheduler["summary"]["max_live_slot_pressure"] == 1.0
    assert (
        vdcores_queue_scheduler["summary"]["scheduler_overhead_ns_mean_per_sm"]
        == 1763558
    )
    assert vdcores_queue_scheduler["summary"]["final_token"] == "ref=25, dae=25"
    tk_official_attempt = attempts_by_id[
        "thunderkittens_mha_h100_official_benchmark_h200"
    ]
    assert tk_official_attempt["status"] == "partial"
    assert tk_official_attempt["summary"]["benchmark_completed"]
    assert tk_official_attempt["summary"]["correctness_completed"]
    assert tk_official_attempt["summary"]["tk_rows_completed"]
    assert not tk_official_attempt["summary"]["fa3_available"]
    assert tk_official_attempt["summary"]["pytorch_reference_oom"]
    assert "non-MHA ThunderKittens kernels" in tk_official_attempt["blocker"]
    for attempt in execution_attempts:
        assert attempt["artifact_root"].startswith("tmp/")
        assert (ROOT / attempt["artifact_root"]).is_dir()
        assert attempt["artifacts"]
        assert any(path.endswith(".json") for path in attempt["artifacts"])
        for path in attempt["artifacts"]:
            assert path.startswith("tmp/")
            assert (ROOT / path).is_file()
    command_plan_records = serving_command_plan["serving_command_plans"]
    assert serving_command_plan["metadata"]["model_tier"] == "primary"
    assert len(command_plan_records) == 35
    command_plan_run_ids = {
        item["paper_baseline_run_id"] for item in command_plan_records
    }
    assert {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_decode_attention_tile",
    } <= command_plan_run_ids
    command_plan_baselines = {
        item["paper_baseline_id"] for item in command_plan_records
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= command_plan_baselines
    for item in command_plan_records:
        assert item["serving_workload_id"] in serving_by_id
        assert item["batch_size"] in serving_by_id[item["serving_workload_id"]][
            "decode_policy"
        ]["batch_sizes"]
        assert item["prompt_tokens"] == serving_by_id[item["serving_workload_id"]][
            "prompt_policy"
        ]["target_prompt_tokens"]
        assert item["decode_tokens"] == serving_by_id[item["serving_workload_id"]][
            "decode_policy"
        ]["decode_tokens"]
        assert item["commands"]
        assert any(command.get("raw_artifact") for command in item["commands"])
        for command in item["commands"]:
            assert command["kind"]
            assert command["command"]
            if "raw_artifact" in command:
                assert command["raw_artifact"].startswith("tmp/")
    planned_run_ids = {
        item["id"]
        for item in paper_baseline_runs["paper_baseline_runs"]
        if item.get("status", "planned_not_run") != "imported_to_viewer"
    }
    assert planned_run_ids <= set(readiness_by_run)
    for item in readiness_by_run.values():
        assert item["latest_artifact_root"].startswith("tmp/")
        assert (ROOT / item["latest_artifact_root"]).is_dir()
        assert item["latest_status"] in {"pass", "partial", "fail"}
        assert item["checks"]
        assert isinstance(item["blocking_gaps"], list)
        if item["latest_status"] != "pass":
            assert item["blocking_gaps"]
    assert readiness_by_run["mpk_qwen3_native_vs_persistent"][
        "latest_status"
    ] == "pass"
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "mpk_qwen3_native_vs_persistent"
        ]["blocking_gaps"]
    )
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "mpk_persistent_scheduler_trace"
        ]["blocking_gaps"]
    )
    assert readiness_by_run["mpk_persistent_scheduler_trace"][
        "latest_status"
    ] == "pass"
    assert not any(
        "dae.runtime" in gap
        for gap in readiness_by_run[
            "vdcores_resource_policy_trace"
        ]["blocking_gaps"]
    )
    assert not any(
        "HF_TOKEN" in gap
        for gap in readiness_by_run[
            "vdcores_resource_policy_trace"
        ]["blocking_gaps"]
    )
    assert readiness_by_run["vdcores_resource_policy_trace"][
        "latest_status"
    ] == "pass"
    assert any(
        "python_module failed: vllm" in gap
        for gap in readiness_by_run[
            "vllm_serving_and_throughput"
        ]["blocking_gaps"]
    )
    assert any(
        "sglang.bench_serving" in gap
        for gap in readiness_by_run[
            "sglang_serving_and_offline"
        ]["blocking_gaps"]
    )

    matrix_ids = {
        item["id"] for item in paper_evaluation["paper_evaluation_matrix"]
    }
    assert {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    } <= matrix_ids
    covered_baselines = {
        baseline_id
        for item in paper_evaluation["paper_evaluation_matrix"]
        for baseline_id in item["paper_baseline_ids"]
    }
    assert {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    } <= covered_baselines
    for item in paper_evaluation["paper_evaluation_matrix"]:
        assert item["claim"]
        assert item["status"]
        assert item["workload_ids"]
        assert item["method_ids"]
        assert item["hardware_targets"]
        assert "correctness" in item["required_metrics"]
        assert "raw_artifacts" in item["required_metrics"]
        assert item["current_evidence_refs"]
        assert isinstance(item["missing_evidence"], list)
        assert item["promotion_gate"]
        if item["id"] == "host_schedule_launch_overhead":
            assert item["status"] == "ready_for_paper_claim"
            assert "host_schedule_stream_concurrency" in item["workload_ids"]
            assert {
                "pto_stream_serial",
                "pto_stream_parallel",
            } <= set(item["method_ids"])
            stream_refs = [
                ref
                for ref in item["current_evidence_refs"]
                if ref.get("benchmark_id") == "host_schedule_stream_concurrency"
            ]
            assert {
                (ref["gpu"], ref["method_id"]) for ref in stream_refs
            } == {
                ("A100", "pto_stream_serial"),
                ("A100", "pto_stream_parallel"),
                ("H200", "pto_stream_serial"),
                ("H200", "pto_stream_parallel"),
            }
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/combined-stream-pool6-02bca4df/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/graph-replay-sweep-01e30e99/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "tmp/cuda-backend/direct-launch-sweep-626b8c75/"
                for ref in item["current_evidence_refs"]
            )
            assert item["missing_evidence"] == []
        if item["id"] == "persistent_device_scheduler_overhead":
            assert any(
                ref.get("kind") == "raw_artifact"
                and ref.get("path")
                == "tmp/cuda-backend/scheduler-breakdown-6f7a1040/persistent-scheduler-breakdown-6f7a1040/"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "graph_layered_cross"
                and ref.get("method_id") == "mpk"
                and ref.get("gpu") == "H200"
                for ref in item["current_evidence_refs"]
            )
            assert not any(
                "Scheduler-overhead breakdown" in gap
                for gap in item["missing_evidence"]
            )
            assert not any(
                "MPK bounded-decode run" in gap
                for gap in item["missing_evidence"]
            )
        if item["id"] == "llm_serving_paper_baselines":
            assert not any(
                "Selected shared model" in gap
                for gap in item["missing_evidence"]
            )
            assert not any(
                "PTO CUDA serving workload" in gap
                for gap in item["missing_evidence"]
            )
            assert any(
                ref.get("path")
                == "docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("path")
                == "docs/nvidia-backend/benchmark-viewer/data/serving_command_plan.json"
                for ref in item["current_evidence_refs"]
            )
            assert any(
                ref.get("kind") == "viewer_result"
                and ref.get("benchmark_id") == "llm_serving_decode"
                and ref.get("method_id") == "pto_persistent_device"
                and ref.get("gpu") == "H200"
                for ref in item["current_evidence_refs"]
            )

    assert paper_readiness_audit["overall_status"] == "not_paper_ready"
    assert paper_readiness_audit["ready_claims"] == 1
    assert paper_readiness_audit["blocked_claims"] == 3
    assert paper_readiness_audit["claim_audits"]
    for item in paper_readiness_audit["claim_audits"]:
        assert item["matrix_status"]
        assert "paper_baseline_run_readiness_statuses" in item
        if item["id"] == "persistent_device_scheduler_overhead":
            assert item["execution_attempt_statuses"] == []
            assert not any(
                action.get("source") == "execution_attempt"
                and action.get("paper_baseline_run_id")
                == "vdcores_resource_policy_trace"
                for action in item["next_actions"]
            )
            assert not any(
                "VDCores queue-pressure and scheduler-overhead"
                in action["action"]
                for action in item["next_actions"]
            )
        if item["id"] == "host_schedule_launch_overhead":
            assert item["matrix_status"] == "ready_for_paper_claim"
            assert item["ready_for_paper_claim"] is True
            assert item["blockers"] == []
        elif item["id"] == "persistent_device_scheduler_overhead":
            assert item["ready_for_paper_claim"] is False
            assert any(
                "stable baseline instrumentation mode" in blocker
                for blocker in item["blockers"]
            )
        else:
            assert item["ready_for_paper_claim"] is False
            assert item["blockers"]
        assert item["promotion_gate"]

    assert results["snapshot"]["commit"] == "743709f3"
    assert results["snapshot"]["full_capture"]["samples"] == 1350
    assert results["snapshot"]["compact_capture"]["samples"] == 108
    for capture in [
        results["snapshot"]["full_capture"],
        results["snapshot"]["compact_capture"],
    ]:
        artifact_root = ROOT / capture["artifact_root"]
        assert artifact_root.is_dir()
        assert any(path.suffix == ".json" for path in artifact_root.iterdir())
    assert results["result_records"]
    mpk_scheduler_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "graph_layered_cross"
        and record["method_id"] == "mpk"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"] == "tmp/cuda-backend/paper-baselines/mpk/"
    ]
    assert len(mpk_scheduler_records) == 1
    mpk_statistic = mpk_scheduler_records[0]["statistic"]
    assert mpk_scheduler_records[0]["correctness"] == "pass"
    assert mpk_statistic["kind"] == "paper_baseline_scheduler_trace"
    assert mpk_statistic["sample_count"] == 1
    assert mpk_statistic["task_count"] == 7261
    assert mpk_statistic["scheduler_count"] == 16
    assert mpk_statistic["worker_count"] == 128
    assert mpk_statistic["scheduler_overhead_ns"] == 29008768
    assert mpk_statistic["dispatch_trace"]["scheduler_slice_pairs"] == 74792
    assert mpk_statistic["resource_policy"]["split_worker_scheduler"] is True
    assert (
        mpk_statistic["queue_pressure"]["worker_coverage_policy"]
        == "128 workers over 16 local schedulers, eight workers per scheduler "
        "under MAX_WORKER_PER_SCHEDULER=9"
    )
    assert mpk_statistic["generated_kernel_metadata"]["tensor_count"] == 371
    vdcores_scheduler_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "vdcores"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-queue-scheduler-46872fa4/"
    ]
    assert len(vdcores_scheduler_records) == 1
    vdcores_statistic = vdcores_scheduler_records[0]["statistic"]
    assert vdcores_scheduler_records[0]["correctness"] == "pass"
    assert vdcores_statistic["kind"] == "paper_baseline_scheduler_trace"
    assert vdcores_statistic["sample_count"] == 5
    assert vdcores_statistic["scheduler_overhead_ns"] == 1763558
    assert vdcores_statistic["queue_pressure"]["max_live_slot_occupancy"] == 24
    assert vdcores_statistic["queue_pressure"]["max_live_slot_pressure"] == 1.0
    assert vdcores_statistic["dispatch_trace"]["queue_ids"] == [
        "m2c",
        "m2ld0",
        "m2ld1",
        "c2m",
    ]
    assert vdcores_statistic["resource_policy"]["virtual_cores"] == 8
    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "thunderkittens"
        and record["hardware"]["gpu"] == "H200"
        and record["correctness"] == "pass"
        and record["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-67c5c655/"
        for record in results["result_records"]
    )
    thunderkittens_capture_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "thunderkittens"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/"
    ]
    assert len(thunderkittens_capture_records) == 2
    assert {
        record["statistic"]["sample_count"]
        for record in thunderkittens_capture_records
    } == {20}
    assert {
        record["inputs"]["shape"] for record in thunderkittens_capture_records
    } == {
        "mha_h100,b=1,h=1,n=768,d=64,causal=True",
        "mha_h100,b=1,h=4,n=1536,d=64,causal=True",
    }
    thunderkittens_full_sweep_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "thunderkittens"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/thunderkittens/full-sweep-4277aa73/"
    ]
    assert len(thunderkittens_full_sweep_records) == 2
    assert {
        record["statistic"]["kind"] for record in thunderkittens_full_sweep_records
    } == {"paper_baseline_full_sweep_capture"}
    assert {
        record["statistic"]["sample_count"]
        for record in thunderkittens_full_sweep_records
    } == {20}
    assert all(
        record["statistic"]["throughput"] > 0
        and record["statistic"]["attention_flops"] > 0
        and record["statistic"]["max_abs_error"] > 0
        and record["correctness"] == "pass"
        for record in thunderkittens_full_sweep_records
    )
    assert {
        record["inputs"]["shape"] for record in thunderkittens_full_sweep_records
    } == {
        "mha_h100_full_sweep,b=1,h=1,n=768,d=64,causal=True",
        "mha_h100_full_sweep,b=1,h=4,n=1536,d=64,causal=True",
    }
    thunderkittens_serving_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "thunderkittens"
        and record["hardware"]["gpu"] == "H200"
    ]
    assert len(thunderkittens_serving_records) == 5
    assert {
        record["statistic"]["kind"] for record in thunderkittens_serving_records
    } == {"paper_baseline_serving_tile_capture"}
    assert {
        record["statistic"]["sample_count"]
        for record in thunderkittens_serving_records
    } == {20}
    assert {
        record["statistic"]["batch_size"]
        for record in thunderkittens_serving_records
    } == {1, 2, 4, 8, 16}
    assert {
        record["statistic"]["prompt_tokens"]
        for record in thunderkittens_serving_records
    } == {128}
    assert {
        record["statistic"]["decode_tokens"]
        for record in thunderkittens_serving_records
    } == {64}
    assert {
        record["raw_artifact"] for record in thunderkittens_serving_records
    } == {
        "tmp/cuda-backend/paper-baselines/serving-runs/thunderkittens/"
        "vdcores_offline_decode/"
    }
    assert {
        record["inputs"]["shape"] for record in thunderkittens_serving_records
    } == {
        "vdcores_offline_decode,mha_h100,b=1,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64",
        "vdcores_offline_decode,mha_h100,b=2,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64",
        "vdcores_offline_decode,mha_h100,b=4,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64",
        "vdcores_offline_decode,mha_h100,b=8,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64",
        "vdcores_offline_decode,mha_h100,b=16,h=1,n=256,d=64,causal=True,prompt_tokens=128,decode_tokens=64",
    }
    assert all(
        record["statistic"]["throughput_tokens_per_s"] > 0
        and record["correctness"] == "pass"
        for record in thunderkittens_serving_records
    )
    driver_graph_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "host_schedule_vector_ops"
        and record["method_id"] == "direct_driver_graph"
        and record["hardware"]["gpu"] == "A100"
        and record["raw_artifact"] == "tmp/cuda-backend/host-launch-a100-8b6cdaee/"
    ]
    assert len(driver_graph_records) == 1
    assert driver_graph_records[0]["statistic"]["sample_count"] == 10
    assert driver_graph_records[0]["correctness"] == "pass"
    driver_launch_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "host_schedule_vector_ops"
        and record["method_id"] == "direct_driver"
        and record["hardware"]["gpu"] == "A100"
        and record["raw_artifact"] == "tmp/cuda-backend/host-launch-a100-8b6cdaee/"
    ]
    assert len(driver_launch_records) == 1
    assert driver_launch_records[0]["statistic"]["sample_count"] == 10
    assert driver_launch_records[0]["correctness"] == "pass"
    runtime_launch_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "host_schedule_vector_ops"
        and record["method_id"] == "direct_runtime"
        and record["hardware"]["gpu"] == "A100"
        and record["raw_artifact"]
        == "tmp/cuda-backend/host-launch-runtime-a100-e429c07b/"
    ]
    assert len(runtime_launch_records) == 1
    assert runtime_launch_records[0]["statistic"]["sample_count"] == 10
    assert runtime_launch_records[0]["correctness"] == "pass"
    h200_host_launch_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "host_schedule_vector_ops"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"] == "tmp/cuda-backend/host-launch-h200-ec8f272e/"
    ]
    assert {
        record["method_id"] for record in h200_host_launch_records
    } == {
        "pto_host_schedule",
        "direct_runtime",
        "direct_driver",
        "direct_driver_graph",
    }
    assert {
        record["statistic"]["sample_count"]
        for record in h200_host_launch_records
    } == {10}
    assert all(
        record["statistic"]["host_wall_p90_ns"] >= record["statistic"]["host_wall_ns"]
        and record["statistic"]["device_wall_p90_ns"]
        >= record["statistic"]["device_wall_ns"]
        for record in h200_host_launch_records
    )
    assert {
        record["correctness"] for record in h200_host_launch_records
    } == {"pass"}
    tensor_launch_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"]
        in {"direct_runtime", "direct_driver", "direct_driver_graph"}
        and record["raw_artifact"]
        in {
            "tmp/cuda-backend/tensor-launch-a100-09462d04/",
            "tmp/cuda-backend/tensor-launch-h200-09462d04/",
        }
    ]
    assert {
        (record["hardware"]["gpu"], record["method_id"])
        for record in tensor_launch_records
    } == {
        ("A100", "direct_runtime"),
        ("A100", "direct_driver"),
        ("A100", "direct_driver_graph"),
        ("H200", "direct_runtime"),
        ("H200", "direct_driver"),
        ("H200", "direct_driver_graph"),
    }
    assert {
        record["statistic"]["sample_count"]
        for record in tensor_launch_records
    } == {10}
    assert all(
        record["inputs"]["dtype"] == "float32 naive SGEMM"
        and record["statistic"]["host_wall_p90_ns"]
        >= record["statistic"]["host_wall_ns"]
        and record["statistic"]["device_wall_p90_ns"]
        >= record["statistic"]["device_wall_ns"]
        for record in tensor_launch_records
    )
    graph_replay_sweep_records = [
        record
        for record in results["result_records"]
        if record["method_id"] == "direct_driver_graph"
        and record["raw_artifact"]
        == "tmp/cuda-backend/graph-replay-sweep-01e30e99/"
    ]
    assert {
        (record["hardware"]["gpu"], record["benchmark_id"], record["inputs"]["shape"])
        for record in graph_replay_sweep_records
    } == {
        ("A100", "host_schedule_vector_ops", "n=1024 vector"),
        ("A100", "host_schedule_vector_ops", "n=4096 vector"),
        ("A100", "host_schedule_vector_ops", "n=65536 vector"),
        ("A100", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("A100", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("A100", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
        ("H200", "host_schedule_vector_ops", "n=1024 vector"),
        ("H200", "host_schedule_vector_ops", "n=4096 vector"),
        ("H200", "host_schedule_vector_ops", "n=65536 vector"),
        ("H200", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("H200", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("H200", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
    }
    assert {
        record["statistic"]["sample_count"]
        for record in graph_replay_sweep_records
    } == {10}
    assert all(
        record["inputs"]["repeat_policy"]
        in {
            "10-repeat host-launch capture",
            "10-repeat selected tensor launch capture",
            "10-repeat graph-replay sweep capture",
        }
        and record["statistic"]["host_wall_p90_ns"]
        >= record["statistic"]["host_wall_ns"]
        and record["statistic"]["device_wall_p90_ns"]
        >= record["statistic"]["device_wall_ns"]
        and record["correctness"] == "pass"
        for record in graph_replay_sweep_records
    )
    direct_launch_sweep_records = [
        record
        for record in results["result_records"]
        if record["method_id"] in {"direct_runtime", "direct_driver"}
        and record["raw_artifact"]
        == "tmp/cuda-backend/direct-launch-sweep-626b8c75/"
    ]
    assert {
        (
            record["hardware"]["gpu"],
            record["method_id"],
            record["benchmark_id"],
            record["inputs"]["shape"],
        )
        for record in direct_launch_sweep_records
    } == {
        ("A100", "direct_driver", "host_schedule_vector_ops", "n=1024 vector"),
        ("A100", "direct_driver", "host_schedule_vector_ops", "n=4096 vector"),
        ("A100", "direct_driver", "host_schedule_vector_ops", "n=65536 vector"),
        ("A100", "direct_runtime", "host_schedule_vector_ops", "n=1024 vector"),
        ("A100", "direct_runtime", "host_schedule_vector_ops", "n=4096 vector"),
        ("A100", "direct_runtime", "host_schedule_vector_ops", "n=65536 vector"),
        ("A100", "direct_driver", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("A100", "direct_driver", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("A100", "direct_driver", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
        ("A100", "direct_runtime", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("A100", "direct_runtime", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("A100", "direct_runtime", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
        ("H200", "direct_driver", "host_schedule_vector_ops", "n=1024 vector"),
        ("H200", "direct_driver", "host_schedule_vector_ops", "n=4096 vector"),
        ("H200", "direct_driver", "host_schedule_vector_ops", "n=65536 vector"),
        ("H200", "direct_runtime", "host_schedule_vector_ops", "n=1024 vector"),
        ("H200", "direct_runtime", "host_schedule_vector_ops", "n=4096 vector"),
        ("H200", "direct_runtime", "host_schedule_vector_ops", "n=65536 vector"),
        ("H200", "direct_driver", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("H200", "direct_driver", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("H200", "direct_driver", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
        ("H200", "direct_runtime", "tensor_core_tile", "n=1024, tensor tile 16x16x16"),
        ("H200", "direct_runtime", "tensor_core_tile", "n=4096, tensor tile 16x16x16"),
        ("H200", "direct_runtime", "tensor_core_tile", "n=65536, tensor tile 16x16x16"),
    }
    assert {
        record["statistic"]["sample_count"]
        for record in direct_launch_sweep_records
    } == {10}
    assert all(
        record["inputs"]["repeat_policy"]
        in {
            "10-repeat CUDA Runtime API capture",
            "10-repeat host-launch capture",
            "10-repeat selected tensor launch capture",
            "10-repeat direct-launch sweep capture",
        }
        and record["statistic"]["host_wall_p90_ns"]
        >= record["statistic"]["host_wall_ns"]
        and record["statistic"]["device_wall_p90_ns"]
        >= record["statistic"]["device_wall_ns"]
        and record["correctness"] == "pass"
        for record in direct_launch_sweep_records
    )
    pto_tensor_core_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "pto_persistent_device"
        and record["raw_artifact"]
        == "tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/"
    ]
    assert {
        (record["hardware"]["gpu"], record["inputs"]["shape"])
        for record in pto_tensor_core_records
    } == {
        ("A100", "n=1024, tensor tile 16x16x16"),
        ("H200", "n=1024, tensor tile 16x16x16"),
    }
    assert {
        record["inputs"]["dtype"] for record in pto_tensor_core_records
    } == {"tf32 WMMA tensor-core, f32 accumulator"}
    assert {
        record["statistic"]["sample_count"]
        for record in pto_tensor_core_records
    } == {1}
    assert all(record["correctness"] == "pass" for record in pto_tensor_core_records)
    triton_tensor_core_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "triton"
        and record["raw_artifact"].startswith(
            "tmp/cuda-backend/paper-baselines/triton/tensor-tile-"
        )
    ]
    assert {
        (record["hardware"]["gpu"], record["inputs"]["shape"])
        for record in triton_tensor_core_records
    } == {
        ("A100", "n=1024, tensor tile 16x16x16"),
        ("H200", "n=1024, tensor tile 16x16x16"),
    }
    assert {
        record["inputs"]["dtype"] for record in triton_tensor_core_records
    } == {"tf32 Triton tl.dot, f32 accumulator"}
    assert all(
        record["statistic"]["sample_count"] >= 10
        and record["statistic"]["max_abs_error"] <= 1.0e-3
        and record["correctness"] == "pass"
        for record in triton_tensor_core_records
    )
    cutlass_tensor_core_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "cutlass"
        and record["raw_artifact"].startswith(
            "tmp/cuda-backend/paper-baselines/cutlass/tensor-tile-"
        )
    ]
    assert {
        (record["hardware"]["gpu"], record["inputs"]["shape"])
        for record in cutlass_tensor_core_records
    } == {
        ("A100", "n=1024, tensor tile 16x16x16"),
        ("H200", "n=1024, tensor tile 16x16x16"),
    }
    assert {
        record["inputs"]["dtype"] for record in cutlass_tensor_core_records
    } == {"tf32 CUTLASS Gemm tensor op, f32 accumulator"}
    assert all(
        record["statistic"]["sample_count"] >= 10
        and record["statistic"]["max_abs_error"] <= 1.0e-3
        and record["correctness"] == "pass"
        for record in cutlass_tensor_core_records
    )
    stream_concurrency_records = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "host_schedule_stream_concurrency"
        and record["method_id"] in {"pto_stream_serial", "pto_stream_parallel"}
        and record["raw_artifact"]
        == "tmp/cuda-backend/combined-stream-pool6-02bca4df"
    ]
    assert {
        (record["hardware"]["gpu"], record["method_id"])
        for record in stream_concurrency_records
    } == {
        ("A100", "pto_stream_serial"),
        ("A100", "pto_stream_parallel"),
        ("H200", "pto_stream_serial"),
        ("H200", "pto_stream_parallel"),
    }
    assert {
        record["statistic"]["sample_count"]
        for record in stream_concurrency_records
    } == {10}
    assert all(
        record["inputs"]["shape"] == "two independent n=1 vector kernels"
        and record["inputs"]["repeat_policy"]
        == "10-repeat stream-concurrency capture"
        and record["statistic"]["host_wall_p90_ns"]
        >= record["statistic"]["host_wall_ns"]
        and record["statistic"]["device_wall_p90_ns"]
        >= record["statistic"]["device_wall_ns"]
        and record["correctness"] == "pass"
        for record in stream_concurrency_records
    )
    for record in results["result_records"]:
        assert record["benchmark_id"] in benchmark_ids
        assert record["method_id"] in method_ids
        assert record["hardware"]["gpu"]
        assert record["statistic"]["sample_count"] > 0
        assert record["raw_artifact"].startswith("tmp/")
        raw_artifact = ROOT / record["raw_artifact"]
        assert raw_artifact.exists()
        if raw_artifact.is_dir():
            assert any(path.suffix == ".json" for path in raw_artifact.iterdir())
        else:
            assert raw_artifact.suffix == ".json"
    pto_serving_equivalent = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "pto_persistent_device"
        and record["hardware"]["gpu"] == "H200"
    ]
    assert len(pto_serving_equivalent) == 1
    assert (
        pto_serving_equivalent[0]["inputs"]["shape"]
        == "controlled serving-equivalent: vdcores_offline_decode attention tile proxy, batch=4, prompt_tokens=128, decode_tokens=64"
    )
    assert (
        pto_serving_equivalent[0]["statistic"]["kind"]
        == "pto_controlled_serving_equivalent"
    )
    assert pto_serving_equivalent[0]["statistic"]["throughput_tokens_per_s"] > 0
    vdcores_guarded_bench = [
        record
        for record in results["result_records"]
        if record["benchmark_id"] == "llm_serving_decode"
        and record["method_id"] == "vdcores"
        and record["hardware"]["gpu"] == "H200"
        and record["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/vdcores/"
        "qwen3-1p7b-repeat-guard-bench-f6b16bac/"
    ]
    assert len(vdcores_guarded_bench) == 1
    assert vdcores_guarded_bench[0]["statistic"]["sample_count"] == 5
    assert vdcores_guarded_bench[0]["statistic"]["device_wall_ns"] == 1778528
    assert vdcores_guarded_bench[0]["correctness"] == "pass"
    assert (
        vdcores_guarded_bench[0]["statistic"]["correctness_artifact"]
        == "tmp/cuda-backend/paper-baselines/vdcores/"
        "qwen3-1p7b-repeat-guard-correctness-712f88e8/"
    )
    assert vdcores_guarded_bench[0]["statistic"]["correctness_check_count"] == 17
    assert (
        vdcores_guarded_bench[0]["raw_artifact"]
        == "tmp/cuda-backend/paper-baselines/vdcores/"
        "qwen3-1p7b-repeat-guard-bench-f6b16bac/"
    )
    assert {"A100", "H200"} <= {
        item["gpu"] for item in results["headline_results"]
    }


def test_review_policy_changelog_and_examples_exist():
    assert (ROOT / ".agents" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "coding-guidance.md").is_file()
    assert (ROOT / ".agents" / "templates" / "ultimate-goal.md").is_file()
    assert (ROOT / ".agents" / "rules" / "core-development.md").is_file()
    assert (ROOT / ".agents" / "rules" / "requirements-first.md").is_file()
    assert (ROOT / ".agents" / "rules" / "testing-and-verification.md").is_file()
    assert (ROOT / ".agents" / "rules" / "ultimate-goal-dispatch.md").is_file()
    assert (ROOT / ".agents" / "rules" / "nvidia-backend-review.md").is_file()
    assert (ROOT / ".agents" / "rules" / "remote-evaluation.md").is_file()
    assert (ROOT / ".agents" / "agents" / "code-review" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "agents" / "documentation-sync" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "agents" / "testing" / "AGENT.md").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_cuda_examples.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_remote_evaluation.py").is_file()
    assert (ROOT / ".agents" / "checks" / "validate_nvidia_changelog.py").is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_viewer_export.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_viewer_export.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_results_update.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_run_readiness.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_mha_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "thunderkittens_full_sweep_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "triton_tensor_tile_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cutlass_tensor_tile_capture.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "cuda_scheduler_breakdown.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_probe.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_baseline_pair_probe.py"
    ).is_file()
    assert (
        ROOT
        / ".agents"
        / "skills"
        / "cuda-backend-eval"
        / "scripts"
        / "paper_serving_command_plan.py"
    ).is_file()
    assert (ROOT / ".agents" / "skills" / "git-commit" / "SKILL.md").is_file()
    assert (ROOT / ".agents" / "skills" / "github-pr" / "SKILL.md").is_file()
    assert (DOC_ROOT / "changelog" / "index.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-review-readiness.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-ultimate-goal.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-benchmark-viewer-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-viewer-result-export.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-changelog-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-cuda-example-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-evaluation-matrix.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-remote-evaluation-contract.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-runs.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-importer.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-probes.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-paired-probe.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-thunderkittens-bounded-capture.md"
    ).is_file()
    assert (DOC_ROOT / "changelog" / "2026-05-31-serving-policy.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-serving-command-plan.md"
    ).is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-paired-probe-dependencies.md"
    ).is_file()

    example_root = ROOT / "examples" / "cuda"
    assert (example_root / "README.md").is_file()
    assert (example_root / "manifest.json").is_file()
    assert (example_root / "host_schedule_vector_ops.py").is_file()
    assert (example_root / "persistent_layered_cross.py").is_file()


def test_ultimate_goal_ci_is_manual_only_and_avoids_ascend_jobs():
    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    workflow_paths += sorted((ROOT / ".github" / "workflows").glob("*.yaml"))
    assert workflow_paths

    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "workflow_dispatch:" in workflow
        assert "pull_request:" not in workflow
        assert "pull_request_target:" not in workflow
        assert "merge_group:" not in workflow
        assert "schedule:" not in workflow
        assert "push:" not in workflow
        assert "runs-on: [self-hosted, a2a3]" not in workflow
        assert "runs-on: [self-hosted, a5]" not in workflow
        assert "--platform a2a3" not in workflow
        assert "--platform a5" not in workflow

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "NVIDIA Manual Review" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "nvidia-manual-review:" in workflow

    ci_doc = (ROOT / "docs" / "ci.md").read_text(encoding="utf-8")
    assert "manual-only" in ci_doc
    assert "must not register automatic" in ci_doc
    assert "a2a3/a5 CI" in ci_doc


def test_ultimate_goal_artifacts_define_paper_ready_cuda_path():
    goal_root = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"
    goal_file = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready.md"

    assert goal_file.is_file()
    assert (goal_root / "dispatch_log.md").is_file()
    assert (goal_root / "work_preparation.md").is_file()
    assert (goal_root / "shared_contracts.md").is_file()
    assert (goal_root / "evaluation_plan.md").is_file()
    assert (goal_root / "baseline_survey.md").is_file()

    goal_text = goal_file.read_text(encoding="utf-8")
    for required in [
        "standalone pto-cu",
        "human-reviewable benchmark viewer",
        "MPK",
        "VDCores",
        "remote evaluation fallback",
        "code evidence",
    ]:
        assert required in goal_text

    evaluation_text = (goal_root / "evaluation_plan.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "paper-ready",
        "Mirage Persistent Kernel",
        "VDCores",
        "CUDA Graph",
        "cuBLAS",
        "A100",
        "H200",
    ]:
        assert required in evaluation_text

    contracts_text = (goal_root / "shared_contracts.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "benchmark_id",
        "method_id",
        "paper_baseline_id",
        "evidence_refs",
        "changelog report",
        "source notes",
    ]:
        assert required in contracts_text

    baseline_text = (goal_root / "baseline_survey.md").read_text(
        encoding="utf-8"
    )
    for required in [
        "mirage-project/mirage",
        "vdcores/vdcores",
        "vLLM",
        "SGLang",
        "ThunderKittens",
        "tmp/baselines/mirage-mpk",
        "tmp/baselines/vdcores",
        "tmp/baselines/vllm",
        "tmp/baselines/sglang",
        "tmp/baselines/thunderkittens",
        "bench_serving",
        "bench throughput",
    ]:
        assert required in baseline_text
