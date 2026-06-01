from __future__ import annotations

from .common import *  # noqa: F403


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
    check_dispatch_log_structure()
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
