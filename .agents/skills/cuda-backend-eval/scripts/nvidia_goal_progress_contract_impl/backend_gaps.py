"""Backend remaining-gap criterion for NVIDIA goal progress."""

from __future__ import annotations

from typing import Any, Callable

MakeCriterion = Callable[..., dict[str, Any]]

BACKEND_GAPS = [
    "Close or reclassify every remaining-gap page linked from "
    "docs/nvidia-backend/status.md before claiming backend completion.",
    "Keep paper/evaluation readiness separate from backend implementation "
    "closure until the status archive has no open remaining-gap links.",
]


def backend_gap_criterion(
    make_criterion: MakeCriterion,
    backend_gap_refs: list[str],
) -> dict[str, Any]:
    return make_criterion(
        identifier="backend_implementation_closure",
        title="Backend implementation gaps are explicit",
        status="in_progress",
        summary=(
            "The CUDA backend status archive still lists implementation gaps, "
            "so goal progress must not imply only paper-result work remains."
        ),
        evidence_refs=[
            "docs/nvidia-backend/status.md",
            *backend_gap_refs,
        ],
        verification=[
            "validate_benchmark_viewer_data.py",
            "check_nvidia_review_ready.py",
        ],
        gaps=BACKEND_GAPS,
    )
