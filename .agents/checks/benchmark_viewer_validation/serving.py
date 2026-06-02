from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def command_mentions_artifact(command_text: str, raw_artifact: str) -> bool:
    path = PurePosixPath(raw_artifact)
    return (
        raw_artifact in command_text
        or raw_artifact.removeprefix("tmp/") in command_text
        or (
            path.parent.as_posix() in command_text
            and path.name in command_text
        )
    )


def validate_serving_workloads(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "serving_workloads", "serving workloads")
    serving_ids = check_unique_ids(records, "serving workload")
    required_workloads = {"mpk_offline_decode", "vdcores_offline_decode"}
    if not required_workloads <= serving_ids:
        missing = sorted(required_workloads - serving_ids)
        fail(f"missing serving workloads: {missing}")

    allowed_status = {
        "policy_selected_no_results",
        "partial_controlled_results",
        "captured_raw",
        "imported_to_viewer",
    }
    required_metrics = {"correctness", "raw_artifacts"}
    required_hardware: set[str] = set()
    for record in records:
        owner = f"serving workload {record['id']}"
        for key in ("title", "status"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")
        source = require_dict(record, "paper_source", owner)
        for key in ("paper", "evidence", "notes"):
            require_string(source, key, owner)
        if not logical_data_path_exists(root, source["evidence"]):
            fail(f"{owner} source evidence missing: {source['evidence']}")

        model_policy = require_dict(record, "model_policy", owner)
        for key in (
            "primary_model",
            "bringup_model",
            "fallback_model",
            "selection_reason",
        ):
            require_string(model_policy, key, owner)

        prompt_policy = require_dict(record, "prompt_policy", owner)
        for key in ("prompt_text", "tokenization_rule"):
            require_string(prompt_policy, key, owner)
        prompt_tokens = prompt_policy.get("target_prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            fail(f"{owner} has invalid prompt target")

        decode_policy = require_dict(record, "decode_policy", owner)
        for key in ("traffic_mode", "generation_mode"):
            require_string(decode_policy, key, owner)
        decode_tokens = decode_policy.get("decode_tokens")
        if not isinstance(decode_tokens, int) or decode_tokens <= 0:
            fail(f"{owner} has invalid decode token count")
        batch_sizes = require_list(decode_policy, "batch_sizes", owner)
        for batch_size in batch_sizes:
            if not isinstance(batch_size, int) or batch_size <= 0:
                fail(f"{owner} has invalid batch size")

        hardware_targets = require_list(record, "hardware_targets", owner)
        for hardware in hardware_targets:
            if not isinstance(hardware, str) or not hardware:
                fail(f"{owner} has invalid hardware target")
            required_hardware.add(hardware)
        require_list(record, "baseline_run_ids", owner)
        metrics = set(require_list(record, "required_metrics", owner))
        if not required_metrics <= metrics:
            missing = sorted(required_metrics - metrics)
            fail(f"{owner} missing required metrics: {missing}")
        require_list(record, "current_blockers", owner)
        check_evidence_refs(record, owner, root)

    if "H200" not in required_hardware:
        fail("serving workloads must include H200")
    return serving_ids


def validate_serving_workload_run_refs(
    data: dict[str, Any],
    baseline_run_ids: set[str],
) -> None:
    for record in data["serving_workloads"]:
        owner = f"serving workload {record['id']}"
        for run_id in record["baseline_run_ids"]:
            if run_id not in baseline_run_ids:
                fail(f"{owner} references unknown baseline run: {run_id}")


def validate_serving_command_plan(
    data: dict[str, Any],
    runs: dict[str, Any],
    serving_workloads: dict[str, Any],
) -> None:
    metadata = require_dict(data, "metadata", "serving command plan")
    artifact_root = require_string(
        metadata,
        "artifact_root",
        "serving command plan metadata",
    )
    if not artifact_root.startswith("tmp/"):
        fail("serving command plan artifact_root must be under tmp/")
    if require_string(metadata, "model_tier", "serving command plan") != "primary":
        fail("serving command plan must use primary model tier")
    expected_sources = {
        "evaluations/nvidia/benchmark-viewer/data/serving_workloads.json",
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_runs.json",
    }
    source_files = set(
        require_list(metadata, "source_files", "serving command plan")
    )
    if source_files != expected_sources:
        fail("serving command plan source_files are stale")

    workloads_by_id = {
        record["id"]: record
        for record in serving_workloads["serving_workloads"]
    }
    runs_by_id = {
        record["id"]: record
        for record in runs["paper_baseline_runs"]
    }
    required_keys: set[tuple[str, str, int]] = set()
    for run in runs["paper_baseline_runs"]:
        if run["paper_evaluation_id"] != "llm_serving_paper_baselines":
            continue
        for workload_id in run["serving_workload_ids"]:
            workload = workloads_by_id[workload_id]
            for batch_size in workload["decode_policy"]["batch_sizes"]:
                required_keys.add((run["id"], workload_id, batch_size))

    records = require_list(data, "serving_command_plans", "serving command plans")
    seen_ids: set[str] = set()
    covered_keys: set[tuple[str, str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            fail("serving command plan record is not an object")
        owner = f"serving command plan {record.get('id', '<missing>')}"
        identifier = require_string(record, "id", owner)
        if identifier in seen_ids:
            fail(f"duplicate serving command plan id: {identifier}")
        seen_ids.add(identifier)
        run_id = require_string(record, "paper_baseline_run_id", owner)
        run = runs_by_id.get(run_id)
        if run is None:
            fail(f"{owner} references unknown paper_baseline_run_id: {run_id}")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id != run["paper_baseline_id"]:
            fail(f"{owner} paper_baseline_id disagrees with run")
        workload_id = require_string(record, "serving_workload_id", owner)
        workload = workloads_by_id.get(workload_id)
        if workload is None:
            fail(f"{owner} references unknown serving_workload_id: {workload_id}")
        batch_size = record.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            fail(f"{owner} has invalid batch_size")
        expected_id = f"{run_id}:{workload_id}:batch{batch_size}"
        if identifier != expected_id:
            fail(f"{owner} id should be {expected_id}")
        covered_keys.add((run_id, workload_id, batch_size))
        if batch_size not in workload["decode_policy"]["batch_sizes"]:
            fail(f"{owner} batch_size is not in workload policy")
        if (
            record.get("prompt_tokens")
            != workload["prompt_policy"]["target_prompt_tokens"]
        ):
            fail(f"{owner} prompt_tokens disagree with workload policy")
        if record.get("decode_tokens") != workload["decode_policy"]["decode_tokens"]:
            fail(f"{owner} decode_tokens disagree with workload policy")
        if require_string(record, "model_tier", owner) != metadata["model_tier"]:
            fail(f"{owner} model_tier disagrees with metadata")
        if (
            require_string(record, "model", owner)
            != workload["model_policy"]["primary_model"]
        ):
            fail(f"{owner} model disagrees with workload primary model")
        require_string(record, "traffic_mode", owner)
        commands = require_list(record, "commands", owner)
        raw_artifact_count = 0
        for command in commands:
            if not isinstance(command, dict):
                fail(f"{owner} command entry is not an object")
            require_string(command, "kind", owner)
            require_string(command, "command", owner)
            raw_artifact = command.get("raw_artifact")
            if raw_artifact is None:
                continue
            if not isinstance(raw_artifact, str) or not raw_artifact.startswith(
                "tmp/"
            ):
                fail(f"{owner} raw_artifact must be under tmp/")
            expected_prefix = f"{artifact_root}/{baseline_id}/{workload_id}/"
            if not raw_artifact.startswith(expected_prefix):
                fail(f"{owner} raw_artifact must be under {expected_prefix}")
            if not command_mentions_artifact(command["command"], raw_artifact):
                fail(f"{owner} command does not mention raw_artifact")
            raw_artifact_count += 1
        if raw_artifact_count == 0:
            fail(f"{owner} has no raw_artifact-producing command")

    if required_keys != covered_keys:
        missing = sorted(required_keys - covered_keys)
        extra = sorted(covered_keys - required_keys)
        fail(
            "serving command plan coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )
