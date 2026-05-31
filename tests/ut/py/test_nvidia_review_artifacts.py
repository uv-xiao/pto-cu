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
    persistent_claim = by_id["persistent_device_scheduler_overhead"]
    persistent_run_ids = {
        run["id"] for run in persistent_claim["paper_baseline_run_statuses"]
    }
    assert {
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
    } <= persistent_run_ids
    assert not any(
        "No paper baseline run record is attached" in blocker
        for blocker in persistent_claim["blockers"]
    )


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
    assert len(records) == 30
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
    assert (VIEWER_ROOT / "data" / "serving_workloads.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").is_file()
    assert (VIEWER_ROOT / "data" / "paper_readiness_audit.json").is_file()
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
        "servingWorkloads",
        "serving_workloads",
        "Serving policies",
        "latest_artifact_root",
        "latest_machine_status",
        "paperEvaluation",
        "paper_evaluation_matrix",
        "paperReadinessAudit",
        "paper_readiness_audit",
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
    assert serving_by_id["vdcores_offline_decode"]["prompt_policy"][
        "target_prompt_tokens"
    ] == 128
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
        if item["paper_evaluation_id"] == "llm_serving_paper_baselines":
            assert item["serving_workload_ids"]
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
            "paired-a100-h200-43b927ed/"
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
        if item["paper_baseline_id"] == "sglang":
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
        if item["id"] == "llm_serving_paper_baselines":
            assert not any(
                "Selected shared model" in gap
                for gap in item["missing_evidence"]
            )
            assert any(
                ref.get("path")
                == "docs/nvidia-backend/benchmark-viewer/data/serving_workloads.json"
                for ref in item["current_evidence_refs"]
            )

    assert paper_readiness_audit["overall_status"] == "not_paper_ready"
    assert paper_readiness_audit["ready_claims"] == 1
    assert paper_readiness_audit["blocked_claims"] == 3
    assert paper_readiness_audit["claim_audits"]
    for item in paper_readiness_audit["claim_audits"]:
        assert item["matrix_status"]
        if item["id"] == "host_schedule_launch_overhead":
            assert item["matrix_status"] == "ready_for_paper_claim"
            assert item["ready_for_paper_claim"] is True
            assert item["blockers"] == []
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
        / "thunderkittens_mha_capture.py"
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
