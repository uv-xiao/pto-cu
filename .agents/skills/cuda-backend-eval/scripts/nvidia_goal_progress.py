#!/usr/bin/env python3
"""Generate ultimate-goal progress data from current NVIDIA artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_AUDIT = VIEWER_DATA / "paper_readiness_audit.json"
DEFAULT_WORK_QUEUE = VIEWER_DATA / "paper_readiness_work_queue.json"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"


def fail(message: str) -> None:
    raise SystemExit(f"nvidia goal progress failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if path.name == "paper_evaluation_matrix.json":
        try:
            return load_viewer_json(path)
        except FileNotFoundError:
            fail(f"missing JSON file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def path_exists(path: str) -> bool:
    target = ROOT / path
    if target.exists():
        return True
    return (
        target.suffix == ".json"
        and target.with_suffix("").is_dir()
        and (target.with_suffix("") / "index.json").is_file()
    )


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


def criterion_status(paths: list[str]) -> str:
    return "met" if all(path_exists(path) for path in paths) else "in_progress"


def build_goal_progress(
    *,
    audit: dict[str, Any],
    work_queue: dict[str, Any],
    matrix: dict[str, Any],
    baselines: dict[str, Any],
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
                    "docs/nvidia-backend/benchmark-viewer/index.html",
                    "docs/nvidia-backend/benchmark-viewer/viewer.js",
                    "docs/nvidia-backend/benchmark-viewer/data/benchmarks.json",
                    "docs/nvidia-backend/benchmark-viewer/data/results.json",
                    "docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json",
                    "docs/nvidia-backend/benchmark-viewer/data/paper_readiness_work_queue.json",
                ]
            ),
            summary="Benchmark setup, methods, results, paper audit, and work queue are JSON-backed.",
            evidence_refs=[
                "docs/nvidia-backend/benchmark-viewer/index.html",
                "docs/nvidia-backend/benchmark-viewer/viewer.js",
                "docs/nvidia-backend/benchmark-viewer/data/",
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
                ]
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
                ]
            ),
            summary="Status docs, shared contracts, and review guards tie claims to explicit code or data evidence.",
            evidence_refs=[
                "docs/nvidia-backend/status.md",
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
                ]
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
                ]
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
                "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json",
                "docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix.json",
            ],
            verification=[
                "validate_benchmark_viewer_data.py",
                "test_ultimate_goal_artifacts_define_paper_ready_cuda_path",
            ],
            gaps=[] if baseline_coverage_met else ["Missing required paper baseline coverage."],
        ),
        make_criterion(
            identifier="paper_grade_results",
            title="Final paper-grade results",
            status="met" if paper_ready else "in_progress",
            summary="Final paper-grade status depends on imported A100/H200 raw artifacts and zero audit blockers.",
            evidence_refs=[
                "docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json",
                "docs/nvidia-backend/benchmark-viewer/data/paper_readiness_work_queue.json",
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
                ]
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
            repo_relative(DEFAULT_AUDIT),
            repo_relative(DEFAULT_WORK_QUEUE),
            repo_relative(DEFAULT_MATRIX),
            repo_relative(DEFAULT_BASELINES),
            "docs/in_progress/nvidia_backend_paper_ready.md",
            "docs/in_progress/nvidia_backend_paper_ready/shared_contracts.md",
            "docs/in_progress/nvidia_backend_paper_ready/evaluation_plan.md",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--work-queue", type=Path, default=DEFAULT_WORK_QUEUE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        "--output",
        type=Path,
        default=VIEWER_DATA / "goal_progress.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_goal_progress(
        audit=load_json(args.audit),
        work_queue=load_json(args.work_queue),
        matrix=load_json(args.matrix),
        baselines=load_json(args.baselines),
    )
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
