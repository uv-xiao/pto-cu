import json
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
                "machine": "dasys-h200x8",
                "baseline": "cublas_sgemm_graph",
                "n": 1024,
                "task_count": 1,
                "host_wall_ns": 60,
                "device_wall_ns": 40,
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
    assert host_record["statistic"]["sample_count"] == 2
    assert host_record["statistic"]["host_wall_ns"] == 140
    assert host_record["statistic"]["device_wall_ns"] == 90
    assert host_record["raw_artifact"] == "tmp/cuda-backend/fixture/"
    assert host_record["correctness"] == "pass"

    assert any(
        record["benchmark_id"] == "tensor_core_tile"
        and record["method_id"] == "cublas_sgemm_graph"
        and record["hardware"]["gpu"] == "H200"
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
                ],
                "next_action": "fixture",
            }
        ]
    }
    baselines_path = tmp_path / "paper_baselines.json"
    probes_path = tmp_path / "paper_baseline_probes.json"
    output_path = tmp_path / "probe.json"
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
    ]


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
    assert (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").is_file()
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
        "latest_artifact_root",
        "paperEvaluation",
        "paper_evaluation_matrix",
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
    paper_evaluation = json.loads(
        (VIEWER_ROOT / "data" / "paper_evaluation_matrix.json").read_text(
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

    run_ids = {item["id"] for item in paper_baseline_runs["paper_baseline_runs"]}
    assert {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
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
        assert item["checks"]
        assert item["next_action"]

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
        assert item["missing_evidence"]
        assert item["promotion_gate"]

    assert results["snapshot"]["commit"] == "743709f3"
    assert results["snapshot"]["full_capture"]["samples"] == 1350
    assert results["snapshot"]["compact_capture"]["samples"] == 108
    assert results["result_records"]
    for record in results["result_records"]:
        assert record["benchmark_id"] in benchmark_ids
        assert record["method_id"] in method_ids
        assert record["hardware"]["gpu"]
        assert record["statistic"]["sample_count"] > 0
        assert record["raw_artifact"].startswith("tmp/")
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
        / "paper_baseline_probe.py"
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
