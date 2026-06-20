import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"
REMOVED_EVAL_SCRIPT_RE = re.compile(
    r"cuda_(smoke|persistent_smoke|benchmark)\.py|cuda-backend-eval/scripts"
)
ACTIVE_REVIEW_SURFACES = [
    VIEWER_ROOT / "data" / "benchmarks.json",
    VIEWER_ROOT / "data" / "methods.json",
    VIEWER_ROOT / "data" / "results.json",
    ROOT / "examples" / "cuda" / "README.md",
    ROOT / "examples" / "cuda" / "host_schedule_vector_ops.py",
    ROOT / "examples" / "cuda" / "persistent_layered_cross.py",
    ROOT / "examples" / "cuda" / "persistent_moe_dispatch_combine.py",
]


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

    benchmarks = json.loads(
        (VIEWER_ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8")
    )
    methods = json.loads(
        (VIEWER_ROOT / "data" / "methods.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (VIEWER_ROOT / "data" / "results.json").read_text(encoding="utf-8")
    )

    benchmark_ids = {item["id"] for item in benchmarks["benchmarks"]}
    assert "graph_layered_cross" in benchmark_ids
    assert "tensor_core_tile" in benchmark_ids
    for benchmark in benchmarks["benchmarks"]:
        assert benchmark["description"]
        assert benchmark["math"]
        assert benchmark["code"]
        assert benchmark["run"]["command"]
        assert benchmark["evidence_refs"]

    method_ids = {item["id"] for item in methods["methods"]}
    assert {"pto_host_schedule", "pto_persistent_device", "cublas_sgemm_graph"} <= method_ids

    assert results["snapshot"]["commit"] == "743709f3"
    assert results["snapshot"]["full_capture"]["samples"] == 1350
    assert results["snapshot"]["compact_capture"]["samples"] == 108
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
    assert (ROOT / ".agents" / "skills" / "git-commit" / "SKILL.md").is_file()
    assert (ROOT / ".agents" / "skills" / "github-pr" / "SKILL.md").is_file()
    assert (DOC_ROOT / "changelog" / "index.md").is_file()
    assert (
        DOC_ROOT / "changelog" / "2026-05-31-review-readiness.md"
    ).is_file()

    in_progress_root = ROOT / "docs" / "in_progress" / "nvidia_backend"
    assert (in_progress_root / "vllm_remote_install_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_env_artifact_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_artifact_complete.md").is_file()
    assert (in_progress_root / "vllm_remote_model_load_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_server_health_probe.md").is_file()
    assert (in_progress_root / "vllm_remote_inference_smoke_probe.md").is_file()
    assert (
        in_progress_root / "vllm_remote_response_contract_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_warmup_shape_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_request_shape_variation_probe.md"
    ).is_file()
    assert (
        in_progress_root / "vllm_remote_serving_semantics_probe.md"
    ).is_file()
    assert (
        in_progress_root / "deepseek_v4_flash_serving_readiness.md"
    ).is_file()

    assert (ROOT / "tools" / "check_nvidia_review_ready.py").is_file()

    example_root = ROOT / "examples" / "cuda"
    assert (example_root / "README.md").is_file()
    assert (example_root / "host_schedule_vector_ops.py").is_file()
    assert (example_root / "persistent_layered_cross.py").is_file()
    assert (example_root / "persistent_moe_dispatch_combine.py").is_file()
    assert (example_root / "vllm_deepseek_v4_server_health_probe.py").is_file()
    assert (example_root / "vllm_deepseek_v4_inference_smoke_probe.py").is_file()
    assert (
        example_root / "vllm_deepseek_v4_response_contract_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_warmup_shape_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_request_shape_variation_probe.py"
    ).is_file()
    assert (
        example_root / "vllm_deepseek_v4_serving_semantics_probe.py"
    ).is_file()


def test_active_review_surfaces_do_not_reference_removed_eval_scripts():
    for path in ACTIVE_REVIEW_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert REMOVED_EVAL_SCRIPT_RE.search(text) is None, path


def test_nvidia_branch_ci_avoids_ascend_jobs():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "nvidia-review:" in workflow
    assert "runs-on: [self-hosted, a2a3]" not in workflow
    assert "runs-on: [self-hosted, a5]" not in workflow
    assert "--platform a2a3" not in workflow
    assert "--platform a5" not in workflow
