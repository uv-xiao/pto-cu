from __future__ import annotations

from .common import *  # noqa: F403


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
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py",
        ".agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py",
        ".agents/skills/cuda-backend-eval/scripts/vdcores_instruction_window_plan.py",
        ".agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py",
        ".agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py",
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
            "No runnable workflow YAML",
            "closed-CI policy",
            "a2a3/a5 CI",
        ],
    )
    workflow_paths = sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    )
    if workflow_paths:
        relpaths = [str(path.relative_to(ROOT)) for path in workflow_paths]
        fail(f"GitHub workflow YAML must stay closed during ultimate goal: {relpaths}")
    workflow = ARCHIVED_WORKFLOW.read_text(encoding="utf-8")
    required_text = [
        "NVIDIA Manual Review",
        "workflow_dispatch:",
        "nvidia-manual-review:",
    ]
    for needle in required_text:
        if needle not in workflow:
            fail(
                "docs/ci/nvidia-manual-review.workflow.yml missing required "
                f"text: {needle}"
            )
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
            fail(
                "docs/ci/nvidia-manual-review.workflow.yml must not contain: "
                f"{needle}"
            )

