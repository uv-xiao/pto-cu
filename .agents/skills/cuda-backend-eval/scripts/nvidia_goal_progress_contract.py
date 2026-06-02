"""Goal-progress criteria for the NVIDIA backend objective."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nvidia_goal_progress_contract_impl.backend_gaps import (
    backend_gap_criterion,
)


PathExists = Callable[[str], bool]
RepoRelative = Callable[[Path], str]


def make_criterion(
    *,
    identifier: str,
    title: str,
    status: str,
    summary: str,
    evidence_refs: list[str],
    verification: list[str],
    gaps: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": identifier,
        "title": title,
        "status": status,
        "summary": summary,
        "evidence_refs": evidence_refs,
        "verification": verification,
        "gaps": gaps,
    }
    if extra:
        payload.update(extra)
    return payload


def criterion_status(paths: list[str], path_exists: PathExists) -> str:
    return "met" if all(path_exists(path) for path in paths) else "in_progress"


def build_goal_progress(
    *,
    audit: dict[str, Any],
    work_queue: dict[str, Any],
    matrix: dict[str, Any],
    baselines: dict[str, Any],
    default_audit: Path,
    default_work_queue: Path,
    default_matrix: Path,
    default_baselines: Path,
    path_exists: PathExists,
    repo_relative: RepoRelative,
    backend_gap_refs: list[str],
) -> dict[str, Any]:
    queue_items = int(work_queue.get("summary", {}).get("total_work_items", 0))
    paper_readiness_status = str(audit.get("overall_status", "unknown"))
    paper_ready = paper_readiness_status == "paper_ready" and queue_items == 0
    baseline_ids = {
        item.get("id")
        for item in baselines.get("paper_baselines", [])
        if isinstance(item, dict)
    }
    matrix_baselines = {
        baseline_id
        for item in matrix.get("paper_evaluation_matrix", [])
        if isinstance(item, dict)
        for baseline_id in item.get("paper_baseline_ids", [])
    }
    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    baseline_coverage_met = (
        required_baselines <= baseline_ids
        and required_baselines <= matrix_baselines
    )

    criteria = [
        make_criterion(
            identifier="benchmark_viewer",
            title="Human-reviewable benchmark viewer",
            status=criterion_status(
                [
                    "evaluations/nvidia/benchmark-viewer/viewer/index.html",
                    "evaluations/nvidia/benchmark-viewer/viewer/viewer.js",
                    "evaluations/nvidia/benchmark-viewer/data/benchmarks.json",
                    "evaluations/nvidia/benchmark-viewer/data/results.json",
                    "evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json",
                    "evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json",
                ],
                path_exists,
            ),
            summary="Benchmark setup, methods, results, paper audit, and work queue are JSON-backed.",
            evidence_refs=[
                "evaluations/nvidia/benchmark-viewer/viewer/index.html",
                "evaluations/nvidia/benchmark-viewer/viewer/viewer.js",
                "evaluations/nvidia/benchmark-viewer/data/",
            ],
            verification=[
                "validate_benchmark_viewer_data.py",
                "test_benchmark_viewer_has_json_backed_review_data",
            ],
            gaps=[],
        ),
        make_criterion(
            identifier="nvidia_examples",
            title="NVIDIA examples match evaluated workloads",
            status=criterion_status(
                [
                    "examples/cuda/manifest.json",
                    "examples/cuda/README.md",
                    "examples/cuda/host_schedule_vector_ops.py",
                    "examples/cuda/persistent_layered_cross.py",
                ],
                path_exists,
            ),
            summary="CUDA examples are declared in a manifest and linked to viewer benchmark/method IDs.",
            evidence_refs=[
                "examples/cuda/manifest.json",
                "examples/cuda/README.md",
            ],
            verification=[
                "validate_cuda_examples.py",
                "test_benchmark_viewer_has_json_backed_review_data",
            ],
            gaps=[],
        ),
        make_criterion(
            identifier="stable_docs_evidence",
            title="Stable docs distinguish implementation from plans",
            status=criterion_status(
                [
                    "docs/nvidia-backend/status.md",
                    "docs/nvidia-backend/overall.md",
                    "docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md",
                    ".agents/checks/check_nvidia_review_ready.py",
                ],
                path_exists,
            ),
            summary="Status docs, shared contracts, and review guards tie claims to explicit code or data evidence.",
            evidence_refs=[
                "docs/nvidia-backend/status.md",
                "docs/nvidia-backend/status/",
                "docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md",
                ".agents/checks/check_nvidia_review_ready.py",
            ],
            verification=[
                "check_nvidia_review_ready.py",
                "validate_benchmark_viewer_data.py",
            ],
            gaps=[],
        ),
        make_criterion(
            identifier="changelog_reports",
            title="Changelog and reporting workflow",
            status=criterion_status(
                [
                    "docs/nvidia-backend/changelog/index.md",
                    ".agents/checks/validate_nvidia_changelog.py",
                ],
                path_exists,
            ),
            summary="Every NVIDIA backend slice is indexed under the changelog and checked for required sections.",
            evidence_refs=[
                "docs/nvidia-backend/changelog/index.md",
                ".agents/checks/validate_nvidia_changelog.py",
            ],
            verification=["validate_nvidia_changelog.py"],
            gaps=[],
        ),
        make_criterion(
            identifier="remote_evaluation",
            title="Remote evaluation fallback policy",
            status=criterion_status(
                [
                    ".agents/checks/validate_remote_evaluation.py",
                    ".agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py",
                    "docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md",
                ],
                path_exists,
            ),
            summary="Remote evaluation supports guarded Git refresh and SSH tree-sync fallback.",
            evidence_refs=[
                ".agents/checks/validate_remote_evaluation.py",
                ".agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py",
            ],
            verification=[
                "validate_remote_evaluation.py",
                "test_paper_baseline_pair_probe_uses_remote_fallback_contract",
            ],
            gaps=[],
        ),
        make_criterion(
            identifier="paper_evaluation_plan",
            title="Paper-ready evaluation plan covers MPK, VDCores, and baselines",
            status="met" if baseline_coverage_met else "in_progress",
            summary="The paper matrix and baseline records cover MPK, VDCores, vLLM, SGLang, and ThunderKittens.",
            evidence_refs=[
                "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan.md",
                "docs/in_progress/nvidia_backend_paper_ready/baseline_survey.md",
                "evaluations/nvidia/benchmark-viewer/data/paper_baselines.json",
                "evaluations/nvidia/benchmark-viewer/data/paper_evaluation_matrix.json",
            ],
            verification=[
                "validate_benchmark_viewer_data.py",
                "test_ultimate_goal_artifacts_define_paper_ready_cuda_path",
            ],
            gaps=[] if baseline_coverage_met else ["Missing required paper baseline coverage."],
        ),
        backend_gap_criterion(make_criterion, backend_gap_refs),
        make_criterion(
            identifier="paper_grade_results",
            title="Final paper-grade results",
            status="met" if paper_ready else "in_progress",
            summary="Final paper-grade status depends on imported A100/H200 raw artifacts and zero audit blockers.",
            evidence_refs=[
                "evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json",
                "evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json",
            ],
            verification=[
                "paper_readiness_audit.py",
                "paper_readiness_work_queue.py",
                "validate_benchmark_viewer_data.py",
            ],
            gaps=[] if paper_ready else [
                "Import the remaining queued paper-readiness artifacts listed "
                "in paper_readiness_work_queue.json.",
                "Promote paper readiness only after the audit has zero blockers.",
            ],
            extra={
                "paper_readiness_status": paper_readiness_status,
                "blocking_work_items": queue_items,
            },
        ),
        make_criterion(
            identifier="dispatcher_log",
            title="Dispatcher log is resumable",
            status=criterion_status(
                [
                    "docs/in_progress/nvidia_backend_paper_ready/dispatch_log.md",
                    "docs/in_progress/nvidia_backend_paper_ready/dispatch_log/index.md",
                    "docs/in_progress/nvidia_backend_paper_ready.md",
                ],
                path_exists,
            ),
            summary="Goal state, branches, PR target, verification, and handoffs are recorded for later sessions in a split dispatch archive.",
            evidence_refs=[
                "docs/in_progress/nvidia_backend_paper_ready.md",
                "docs/in_progress/nvidia_backend_paper_ready/dispatch_log.md",
                "docs/in_progress/nvidia_backend_paper_ready/dispatch_log/index.md",
            ],
            verification=["check_nvidia_review_ready.py"],
            gaps=[],
        ),
    ]

    status_counts: dict[str, int] = {}
    for item in criteria:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    overall_status = "complete" if status_counts.get("in_progress", 0) == 0 else "in_progress"
    return {
        "schema_version": 1,
        "source_files": [
            repo_relative(default_audit),
            repo_relative(default_work_queue),
            repo_relative(default_matrix),
            repo_relative(default_baselines),
            "docs/in_progress/nvidia_backend_paper_ready.md",
            "docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/baseline_families.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/workloads.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/hardware_matrix.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/metrics.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/reproducibility.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/paper_outputs.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan/dispatcher_backlog.md",
            "docs/nvidia-backend/status.md",
            "docs/nvidia-backend/status/remaining-gaps/",
        ],
        "overall_status": overall_status,
        "summary": {
            "criteria_total": len(criteria),
            "criteria_met": status_counts.get("met", 0),
            "criteria_in_progress": status_counts.get("in_progress", 0),
            "criteria_by_status": dict(sorted(status_counts.items())),
        },
        "acceptance_criteria": criteria,
    }
