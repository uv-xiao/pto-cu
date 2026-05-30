#!/usr/bin/env python3
"""Check review-facing CUDA backend docs, viewer data, and examples."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"
GOAL_ROOT = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOW_ROOT / "ci.yml"


def fail(message: str) -> None:
    raise SystemExit(f"nvidia review guard failed: {message}")


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing file: {path.relative_to(ROOT)}")


def load_json(path: Path) -> dict:
    require_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")


def check_evaluation_docs() -> None:
    root_docs = sorted(DOC_ROOT.glob("evaluation*.md"))
    names = {path.name for path in root_docs}
    if names != {"evaluation.md", "evaluation-current.md"}:
        fail(f"unexpected root evaluation docs: {sorted(names)}")
    for path in root_docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 220:
            fail(f"{path.relative_to(ROOT)} has {len(lines)} lines")
    require_file(DOC_ROOT / "history" / "index.md")
    require_file(DOC_ROOT / "history" / "captures" / "current-head-layered-cross-743709f3.md")
    require_file(DOC_ROOT / "history" / "captures" / "legacy-captures.md")
    require_file(DOC_ROOT / "changelog" / "index.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-review-readiness.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-ultimate-goal.md")
    require_file(
        DOC_ROOT / "changelog" / "2026-05-31-benchmark-viewer-contract.md"
    )
    require_file(DOC_ROOT / "changelog" / "2026-05-31-viewer-result-export.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-changelog-contract.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-cuda-example-contract.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-paper-evaluation-matrix.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-remote-evaluation-contract.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-runs.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-importer.md")
    require_file(DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-probes.md")
    require_file(
        DOC_ROOT / "changelog" / "2026-05-31-paper-baseline-paired-probe.md"
    )
    require_file(
        DOC_ROOT / "changelog" / "2026-05-31-thunderkittens-dependency-probe.md"
    )
    require_file(DOC_ROOT / "changelog" / "2026-05-31-thunderkittens-quick-smoke.md")
    require_file(
        DOC_ROOT / "changelog" / "2026-05-31-thunderkittens-bounded-capture.md"
    )
    require_file(DOC_ROOT / "changelog" / "2026-05-31-serving-policy.md")


def require_text(path: Path, needles: list[str]) -> None:
    require_file(path)
    text = path.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            fail(f"{path.relative_to(ROOT)} missing required text: {needle}")


def check_ultimate_goal_contract() -> None:
    require_text(
        ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready.md",
        [
            "standalone pto-cu",
            "human-reviewable benchmark viewer",
            "MPK",
            "VDCores",
            "remote evaluation fallback",
            "code evidence",
        ],
    )
    require_file(GOAL_ROOT / "dispatch_log.md")
    require_file(GOAL_ROOT / "work_preparation.md")
    require_text(
        GOAL_ROOT / "baseline_survey.md",
        [
            "mirage-project/mirage",
            "vdcores/vdcores",
            "vLLM",
            "SGLang",
            "ThunderKittens",
            "serving_workloads.json",
            "mpk_offline_decode",
            "vdcores_offline_decode",
            "tmp/baselines/mirage-mpk",
            "tmp/baselines/vdcores",
            "tmp/baselines/vllm",
            "tmp/baselines/sglang",
            "tmp/baselines/thunderkittens",
            "bench_serving",
            "bench throughput",
        ],
    )
    require_text(
        GOAL_ROOT / "shared_contracts.md",
        [
            "benchmark_id",
            "method_id",
            "paper_baseline_id",
            "evidence_refs",
            "changelog report",
            "source notes",
        ],
    )
    require_text(
        GOAL_ROOT / "evaluation_plan.md",
        [
            "paper-ready",
            "Mirage Persistent Kernel",
            "VDCores",
            "CUDA Graph",
            "cuBLAS",
            "A100",
            "H200",
        ],
    )


def check_evidence_refs(records: list[dict], owner: str) -> None:
    for record in records:
        for ref in record.get("evidence_refs", []):
            path = ROOT / ref["path"]
            require_file(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            for symbol in ref.get("symbols", []):
                if symbol not in text:
                    fail(f"{owner} {record['id']} missing symbol {symbol} in {ref['path']}")


def check_viewer_data() -> None:
    require_file(VIEWER_ROOT / "index.html")
    require_file(VIEWER_ROOT / "styles.css")
    require_file(VIEWER_ROOT / "viewer.js")
    viewer_script = (VIEWER_ROOT / "viewer.js").read_text(encoding="utf-8")
    for needle in [
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
        "latest_artifact_root",
        "paperEvaluation",
        "paper_evaluation_matrix",
        "result_records",
        "raw_artifact",
        "correctness",
    ]:
        if needle not in viewer_script:
            fail(f"viewer.js does not render contract field: {needle}")

    benchmarks = load_json(VIEWER_ROOT / "data" / "benchmarks.json")
    methods = load_json(VIEWER_ROOT / "data" / "methods.json")
    capture_imports = load_json(VIEWER_ROOT / "data" / "capture_imports.json")
    paper_baselines = load_json(VIEWER_ROOT / "data" / "paper_baselines.json")
    paper_baseline_runs = load_json(
        VIEWER_ROOT / "data" / "paper_baseline_runs.json"
    )
    paper_baseline_probes = load_json(
        VIEWER_ROOT / "data" / "paper_baseline_probes.json"
    )
    serving_workloads = load_json(VIEWER_ROOT / "data" / "serving_workloads.json")
    paper_evaluation = load_json(
        VIEWER_ROOT / "data" / "paper_evaluation_matrix.json"
    )
    results = load_json(VIEWER_ROOT / "data" / "results.json")

    benchmark_ids = {item["id"] for item in benchmarks.get("benchmarks", [])}
    required_benchmarks = {
        "host_schedule_vector_ops",
        "graph_layered_cross",
        "tensor_core_tile",
        "llm_serving_decode",
    }
    if not required_benchmarks <= benchmark_ids:
        fail(f"missing benchmark ids: {sorted(required_benchmarks - benchmark_ids)}")
    for benchmark in benchmarks["benchmarks"]:
        for key in ("description", "math", "code"):
            if not benchmark.get(key):
                fail(f"benchmark {benchmark['id']} has empty {key}")
        if not benchmark.get("run", {}).get("command"):
            fail(f"benchmark {benchmark['id']} has no run command")
    check_evidence_refs(benchmarks["benchmarks"], "benchmark")

    method_ids = {item["id"] for item in methods.get("methods", [])}
    required_methods = {
        "pto_host_schedule",
        "pto_persistent_device",
        "direct_driver_graph",
        "cublas_sgemm_graph",
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    }
    if not required_methods <= method_ids:
        fail(f"missing method ids: {sorted(required_methods - method_ids)}")
    check_evidence_refs(methods["methods"], "method")

    import_rules = capture_imports.get("capture_imports", [])
    import_baselines = {item["baseline"] for item in import_rules}
    required_import_baselines = {
        "pto_host_schedule",
        "pto_persistent_dag_graph_layered_cross",
        "cublas_sgemm_graph",
    }
    if not required_import_baselines <= import_baselines:
        missing = sorted(required_import_baselines - import_baselines)
        fail(f"missing capture import baselines: {missing}")

    paper_baseline_ids = {
        item["id"] for item in paper_baselines.get("paper_baselines", [])
    }
    required_paper_baselines = {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    }
    if not required_paper_baselines <= paper_baseline_ids:
        missing = sorted(required_paper_baselines - paper_baseline_ids)
        fail(f"missing paper baseline ids: {missing}")
    for baseline in paper_baselines["paper_baselines"]:
        for key in ("status", "paper_role", "next_action"):
            if not baseline.get(key):
                fail(f"paper baseline {baseline['id']} has empty {key}")
        source = baseline.get("source", {})
        if not source.get("upstream_url") or not source.get("local_tmp_path"):
            fail(f"paper baseline {baseline['id']} has incomplete source")
        if baseline["status"] != "source_cloned_for_survey":
            fail(f"paper baseline {baseline['id']} is not cloned for survey")
        if len(source.get("commit", "")) != 40:
            fail(f"paper baseline {baseline['id']} has no pinned commit")

    serving_workload_ids = {
        item["id"] for item in serving_workloads.get("serving_workloads", [])
    }
    required_serving_workloads = {"mpk_offline_decode", "vdcores_offline_decode"}
    if not required_serving_workloads <= serving_workload_ids:
        missing = sorted(required_serving_workloads - serving_workload_ids)
        fail(f"missing serving workload ids: {missing}")
    check_evidence_refs(serving_workloads["serving_workloads"], "serving workload")

    run_ids = {
        item["id"] for item in paper_baseline_runs.get("paper_baseline_runs", [])
    }
    required_run_ids = {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
    }
    if not required_run_ids <= run_ids:
        missing = sorted(required_run_ids - run_ids)
        fail(f"missing paper baseline run ids: {missing}")

    probe_baseline_ids = {
        item["paper_baseline_id"]
        for item in paper_baseline_probes.get("paper_baseline_probes", [])
    }
    if not required_paper_baselines <= probe_baseline_ids:
        missing = sorted(required_paper_baselines - probe_baseline_ids)
        fail(f"missing paper baseline probe coverage: {missing}")

    matrix_ids = {
        item["id"] for item in paper_evaluation.get("paper_evaluation_matrix", [])
    }
    required_matrix_ids = {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    }
    if not required_matrix_ids <= matrix_ids:
        missing = sorted(required_matrix_ids - matrix_ids)
        fail(f"missing paper evaluation matrix ids: {missing}")
    matrix_baselines = {
        baseline_id
        for item in paper_evaluation["paper_evaluation_matrix"]
        for baseline_id in item.get("paper_baseline_ids", [])
    }
    if not required_paper_baselines <= matrix_baselines:
        missing = sorted(required_paper_baselines - matrix_baselines)
        fail(f"paper evaluation matrix missing baseline ids: {missing}")

    snapshot = results.get("snapshot", {})
    if snapshot.get("commit") != "743709f3":
        fail("viewer snapshot commit must be 743709f3")
    if snapshot.get("full_capture", {}).get("samples") != 1350:
        fail("viewer full capture sample count must be 1350")
    if snapshot.get("compact_capture", {}).get("samples") != 108:
        fail("viewer compact capture sample count must be 108")


def check_viewer_schema_contract() -> None:
    validator_path = (
        ROOT / ".agents" / "checks" / "validate_benchmark_viewer_data.py"
    )
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_benchmark_viewer_data", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load benchmark viewer data validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_viewer_data(ROOT)


def check_changelog_contract() -> None:
    validator_path = (
        ROOT / ".agents" / "checks" / "validate_nvidia_changelog.py"
    )
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_nvidia_changelog", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load NVIDIA changelog validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_changelog(ROOT)


def check_cuda_example_contract() -> None:
    validator_path = ROOT / ".agents" / "checks" / "validate_cuda_examples.py"
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_cuda_examples", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load CUDA example validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_examples(ROOT)


def check_remote_evaluation_contract() -> None:
    validator_path = ROOT / ".agents" / "checks" / "validate_remote_evaluation.py"
    require_file(validator_path)
    spec = importlib.util.spec_from_file_location(
        "validate_remote_evaluation", validator_path
    )
    if spec is None or spec.loader is None:
        fail("could not load remote evaluation validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.validate_remote_evaluation(ROOT)


def check_examples_and_rules() -> None:
    for relpath in [
        ".agents/AGENT.md",
        ".agents/coding-guidance.md",
        ".agents/templates/ultimate-goal.md",
        ".agents/rules/core-development.md",
        ".agents/rules/example-requirements.md",
        ".agents/rules/nvidia-backend-review.md",
        ".agents/rules/requirements-first.md",
        ".agents/rules/remote-evaluation.md",
        ".agents/rules/quality-evidence.md",
        ".agents/rules/testing-and-verification.md",
        ".agents/rules/ultimate-goal-dispatch.md",
        ".agents/agents/code-review/AGENT.md",
        ".agents/agents/documentation-sync/AGENT.md",
        ".agents/agents/testing/AGENT.md",
        ".agents/checks/validate_cuda_examples.py",
        ".agents/checks/validate_remote_evaluation.py",
        ".agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py",
        ".agents/checks/validate_nvidia_changelog.py",
        ".agents/skills/git-commit/SKILL.md",
        ".agents/skills/github-pr/SKILL.md",
        "examples/cuda/README.md",
        "examples/cuda/manifest.json",
        "examples/cuda/host_schedule_vector_ops.py",
        "examples/cuda/persistent_layered_cross.py",
    ]:
        require_file(ROOT / relpath)


def check_manual_ci_policy() -> None:
    require_text(
        ROOT / "docs" / "ci.md",
        [
            "manual-only",
            "must not register automatic",
            "a2a3/a5 CI",
        ],
    )
    workflow_paths = sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    )
    if not workflow_paths:
        fail("missing GitHub workflow files")
    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        relpath = workflow_path.relative_to(ROOT)
        if "workflow_dispatch:" not in workflow:
            fail(f"{relpath} must be manual-only and include workflow_dispatch")
        forbidden_text = [
            "pull_request:",
            "pull_request_target:",
            "merge_group:",
            "schedule:",
            "push:",
            "runs-on: [self-hosted, a2a3]",
            "runs-on: [self-hosted, a5]",
            "--platform a2a3",
            "--platform a5",
        ]
        for needle in forbidden_text:
            if needle in workflow:
                fail(f"{relpath} must not contain: {needle}")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    required_text = [
        "NVIDIA Manual Review",
        "nvidia-manual-review:",
    ]
    for needle in required_text:
        if needle not in workflow:
            fail(f".github/workflows/ci.yml missing required text: {needle}")


def main() -> None:
    check_evaluation_docs()
    check_ultimate_goal_contract()
    check_viewer_data()
    check_viewer_schema_contract()
    check_changelog_contract()
    check_cuda_example_contract()
    check_remote_evaluation_contract()
    check_examples_and_rules()
    check_manual_ci_policy()
    print("nvidia review guard passed")


if __name__ == "__main__":
    main()
