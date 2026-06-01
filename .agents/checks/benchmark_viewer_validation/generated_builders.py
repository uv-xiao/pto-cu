"""Load generated-data builders used for freshness checks."""

from __future__ import annotations

import importlib.util

from .common import ROOT, fail


READINESS_AUDIT_SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "cuda-backend-eval"
    / "scripts"
    / "paper_readiness_audit.py"
)
WORK_QUEUE_SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "cuda-backend-eval"
    / "scripts"
    / "paper_readiness_work_queue.py"
)
GOAL_PROGRESS_SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "cuda-backend-eval"
    / "scripts"
    / "nvidia_goal_progress.py"
)


def load_readiness_audit_builder():
    spec = importlib.util.spec_from_file_location(
        "paper_readiness_audit",
        READINESS_AUDIT_SCRIPT,
    )
    if spec is None or spec.loader is None:
        fail("could not load paper_readiness_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_readiness_audit


def load_work_queue_builder():
    spec = importlib.util.spec_from_file_location(
        "paper_readiness_work_queue",
        WORK_QUEUE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        fail("could not load paper_readiness_work_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_work_queue


def load_goal_progress_builder():
    spec = importlib.util.spec_from_file_location(
        "nvidia_goal_progress",
        GOAL_PROGRESS_SCRIPT,
    )
    if spec is None or spec.loader is None:
        fail("could not load nvidia_goal_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_goal_progress
