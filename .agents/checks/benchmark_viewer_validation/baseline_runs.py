from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def validate_paper_baseline_runs(
    data: dict[str, Any],
    baseline_ids: set[str],
    paper_evaluation_ids: set[str],
    serving_workload_ids: set[str],
    root: Path,
) -> None:
    records = require_list(data, "paper_baseline_runs", "paper baseline runs")
    run_ids = check_unique_ids(records, "paper baseline run")
    required_runs = {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_qwen3_8b_decode_preflight",
        "mpk_persistent_scheduler_trace",
        "vdcores_resource_policy_trace",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
        "thunderkittens_full_sweep",
        "thunderkittens_decode_attention_tile",
    }
    if not required_runs <= run_ids:
        missing = sorted(required_runs - run_ids)
        fail(f"missing paper baseline runs: {missing}")

    allowed_status = {
        "planned_not_run",
        "setup_ready",
        "captured_raw",
        "imported_to_viewer",
    }
    required_baseline_coverage = {
        "mpk",
        "vdcores",
        "vllm",
        "sglang",
        "thunderkittens",
    }
    covered_baselines: set[str] = set()

    for record in records:
        owner = f"paper baseline run {record['id']}"
        for key in ("title", "status"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        covered_baselines.add(baseline_id)
        paper_evaluation_id = require_string(record, "paper_evaluation_id", owner)
        if paper_evaluation_id not in paper_evaluation_ids:
            fail(f"{owner} references unknown paper_evaluation_id: {paper_evaluation_id}")
        serving_ids = record.get("serving_workload_ids", [])
        if not isinstance(serving_ids, list):
            fail(f"{owner} serving_workload_ids is not a list")
        if paper_evaluation_id == "llm_serving_paper_baselines" and not serving_ids:
            fail(f"{owner} must reference at least one serving workload")
        for serving_id in serving_ids:
            if serving_id not in serving_workload_ids:
                fail(f"{owner} references unknown serving_workload_id: {serving_id}")

        for key in (
            "hardware_targets",
            "setup_commands",
            "run_commands",
            "expected_artifacts",
            "required_metrics",
        ):
            values = require_list(record, key, owner)
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    fail(f"{owner} has invalid {key} entry")

        workload = require_dict(record, "workload", owner)
        for key in (
            "model",
            "input_policy",
            "output_policy",
            "batch_or_concurrency",
        ):
            require_string(workload, key, owner)

        metrics = set(record["required_metrics"])
        if not {"correctness", "raw_artifacts"} <= metrics:
            fail(f"{owner} must require correctness and raw_artifacts")
        if paper_evaluation_id == "llm_serving_paper_baselines":
            serving_metrics = {
                "model_and_prompt_shape",
                "batch_or_concurrency_policy",
            }
            if not serving_metrics <= metrics:
                missing = sorted(serving_metrics - metrics)
                fail(f"{owner} missing LLM serving required metrics: {missing}")
        for artifact in record["expected_artifacts"]:
            if not artifact.startswith("tmp/"):
                fail(f"{owner} expected artifact must be under tmp/: {artifact}")
            if record["status"] == "imported_to_viewer" and not (
                root / artifact
            ).exists():
                fail(f"{owner} expected artifact path missing: {artifact}")

        import_target = require_dict(record, "import_target", owner)
        if (
            require_string(import_target, "viewer_file", owner)
            != "evaluations/nvidia/benchmark-viewer/data/results.json"
        ):
            fail(f"{owner} import target must be viewer results.json")
        for key in ("result_kind", "notes"):
            require_string(import_target, key, owner)

    if not required_baseline_coverage <= covered_baselines:
        missing = sorted(required_baseline_coverage - covered_baselines)
        fail(f"paper baseline runs missing baseline coverage: {missing}")


def validate_paper_baseline_run_readiness(
    data: dict[str, Any],
    run_ids: set[str],
    required_run_ids: set[str],
    baseline_ids: set[str],
    root: Path,
) -> None:
    records = require_list(
        data,
        "paper_baseline_run_readiness",
        "paper baseline run readiness",
    )
    readiness_ids = check_unique_ids(records, "paper baseline run readiness")
    required_readiness = {f"{run_id}_readiness" for run_id in required_run_ids}
    if not required_readiness <= readiness_ids:
        missing = sorted(required_readiness - readiness_ids)
        fail(f"missing paper baseline run readiness records: {missing}")
    allowed_status = {"pass", "partial", "fail"}
    covered_runs: set[str] = set()
    for record in records:
        owner = f"paper baseline run readiness {record['id']}"
        for key in ("title", "latest_status", "next_action"):
            require_string(record, key, owner)
        if record["latest_status"] not in allowed_status:
            fail(f"{owner} has invalid latest_status: {record['latest_status']}")
        run_id = require_string(record, "paper_baseline_run_id", owner)
        if run_id not in run_ids:
            fail(f"{owner} references unknown paper_baseline_run_id: {run_id}")
        covered_runs.add(run_id)
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        artifact_root = require_string(record, "latest_artifact_root", owner)
        require_current_artifact_path(root, artifact_root, owner)
        checks = require_list(record, "checks", owner)
        for check in checks:
            if not isinstance(check, dict):
                fail(f"{owner} check is not an object")
            for key in ("kind", "status", "why"):
                require_string(check, key, owner)
            if check["status"] not in allowed_status:
                fail(f"{owner} has invalid check status: {check['status']}")
        gaps = record.get("blocking_gaps")
        if not isinstance(gaps, list) or not all(
            isinstance(gap, str) for gap in gaps
        ):
            fail(f"{owner} blocking_gaps is not a list of strings")
        if record["latest_status"] != "pass" and not gaps:
            fail(f"{owner} is not pass but has no blocking_gaps")
    if not required_run_ids <= covered_runs:
        missing = sorted(required_run_ids - covered_runs)
        fail(f"missing run readiness coverage: {missing}")


def validate_paper_baseline_execution_attempts(
    data: dict[str, Any],
    baseline_ids: set[str],
    run_ids: set[str],
    root: Path,
) -> None:
    records = require_list(
        data,
        "paper_baseline_execution_attempts",
        "paper baseline execution attempts",
    )
    check_unique_ids(records, "paper baseline execution attempt")
    allowed_status = {
        "pass",
        "partial",
        "fail",
        "failed_after_kernel_launch",
        "blocked",
        "blocked_before_model_load",
    }
    for record in records:
        owner = f"paper baseline execution attempt {record['id']}"
        for key in ("title", "status", "artifact_root", "command", "observation"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        run_id = require_string(record, "paper_baseline_run_id", owner)
        if run_id not in run_ids:
            fail(f"{owner} references unknown paper_baseline_run_id: {run_id}")
        artifact_root = require_string(record, "artifact_root", owner)
        if not artifact_root.startswith("tmp/"):
            fail(f"{owner} artifact_root must be under tmp/: {artifact_root}")
        if not (root / artifact_root).is_dir():
            fail(f"{owner} artifact_root is missing: {artifact_root}")
        hardware = require_dict(record, "hardware", owner)
        for key in ("gpu", "machine", "compute_target"):
            require_string(hardware, key, owner)
        artifacts = require_list(record, "artifacts", owner)
        has_json_artifact = False
        for artifact in artifacts:
            if not isinstance(artifact, str) or not artifact.startswith("tmp/"):
                fail(f"{owner} artifact must be under tmp/: {artifact}")
            path = root / artifact
            if not path.is_file():
                fail(f"{owner} artifact path missing: {artifact}")
            has_json_artifact = has_json_artifact or path.suffix == ".json"
        if not has_json_artifact:
            fail(f"{owner} must include at least one JSON artifact")
        patch_markers = " ".join(
            str(record.get(key, ""))
            for key in ("command", "observation", "blocker")
        )
        if "local patch" in patch_markers:
            patch_paths = require_list(record, "reproducibility_patches", owner)
            if not patch_paths:
                fail(f"{owner} local patch needs reproducibility_patches")
            for patch_path in patch_paths:
                if (
                    not isinstance(patch_path, str)
                    or not patch_path.startswith("docs/")
                    or not patch_path.endswith(".patch")
                ):
                    fail(
                        f"{owner} reproducibility patch must be docs/*.patch: "
                        f"{patch_path}"
                    )
                if not (root / patch_path).is_file():
                    fail(f"{owner} reproducibility patch missing: {patch_path}")
        require_dict(record, "summary", owner)


