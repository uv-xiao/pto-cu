#!/usr/bin/env python3
"""Check review-facing CUDA backend docs, viewer data, and examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "docs" / "nvidia-backend"
VIEWER_ROOT = DOC_ROOT / "benchmark-viewer"
GOAL_ROOT = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"


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
        GOAL_ROOT / "shared_contracts.md",
        [
            "benchmark_id",
            "method_id",
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

    benchmarks = load_json(VIEWER_ROOT / "data" / "benchmarks.json")
    methods = load_json(VIEWER_ROOT / "data" / "methods.json")
    results = load_json(VIEWER_ROOT / "data" / "results.json")

    benchmark_ids = {item["id"] for item in benchmarks.get("benchmarks", [])}
    required_benchmarks = {
        "host_schedule_vector_ops",
        "graph_layered_cross",
        "tensor_core_tile",
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
    }
    if not required_methods <= method_ids:
        fail(f"missing method ids: {sorted(required_methods - method_ids)}")
    check_evidence_refs(methods["methods"], "method")

    snapshot = results.get("snapshot", {})
    if snapshot.get("commit") != "743709f3":
        fail("viewer snapshot commit must be 743709f3")
    if snapshot.get("full_capture", {}).get("samples") != 1350:
        fail("viewer full capture sample count must be 1350")
    if snapshot.get("compact_capture", {}).get("samples") != 108:
        fail("viewer compact capture sample count must be 108")


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
        ".agents/skills/git-commit/SKILL.md",
        ".agents/skills/github-pr/SKILL.md",
        "examples/cuda/README.md",
        "examples/cuda/host_schedule_vector_ops.py",
        "examples/cuda/persistent_layered_cross.py",
    ]:
        require_file(ROOT / relpath)


def main() -> None:
    check_evaluation_docs()
    check_ultimate_goal_contract()
    check_viewer_data()
    check_examples_and_rules()
    print("nvidia review guard passed")


if __name__ == "__main__":
    main()
