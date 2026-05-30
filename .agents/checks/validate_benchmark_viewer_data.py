#!/usr/bin/env python3
"""Validate the CUDA benchmark viewer's review-facing data contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def fail(message: str) -> None:
    raise SystemExit(f"benchmark viewer data validation failed: {message}")


def load_json(root: Path, name: str) -> dict[str, Any]:
    path = (
        root / "docs" / "nvidia-backend" / "benchmark-viewer" / "data" / name
    )
    if not path.is_file():
        fail(f"missing data file: {path.relative_to(root)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(root)}: {exc}")


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} has empty or missing {key}")
    return value


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def require_list(record: dict[str, Any], key: str, owner: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def validate_id(identifier: str, owner: str) -> None:
    if not ID_RE.fullmatch(identifier):
        fail(f"{owner} id is not stable snake_case: {identifier}")


def check_unique_ids(records: list[dict[str, Any]], owner: str) -> set[str]:
    ids: set[str] = set()
    for record in records:
        identifier = require_string(record, "id", owner)
        validate_id(identifier, owner)
        if identifier in ids:
            fail(f"duplicate {owner} id: {identifier}")
        ids.add(identifier)
    return ids


def check_evidence_refs(record: dict[str, Any], owner: str, root: Path) -> None:
    refs = require_list(record, "evidence_refs", owner)
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} evidence ref is not an object")
        relpath = require_string(ref, "path", owner)
        path = root / relpath
        if not path.is_file():
            fail(f"{owner} evidence path missing: {relpath}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol in require_list(ref, "symbols", owner):
            if not isinstance(symbol, str) or not symbol:
                fail(f"{owner} evidence symbol is empty")
            if symbol not in text:
                fail(f"{owner} missing evidence symbol {symbol} in {relpath}")


def validate_benchmarks(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "benchmarks", "benchmarks")
    benchmark_ids = check_unique_ids(records, "benchmark")
    for record in records:
        owner = f"benchmark {record['id']}"
        for key in ("title", "description", "math", "code"):
            require_string(record, key, owner)
        run = require_dict(record, "run", owner)
        require_string(run, "command", owner)
        inputs = require_dict(run, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)
        check_evidence_refs(record, owner, root)
    return benchmark_ids


def validate_methods(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "methods", "methods")
    method_ids = check_unique_ids(records, "method")
    allowed_categories = {
        "pto_runtime",
        "vendor_baseline",
        "framework_baseline",
        "paper_baseline",
        "diagnostic_baseline",
    }
    for record in records:
        owner = f"method {record['id']}"
        for key in ("name", "runtime_flow", "lifecycle", "launch_model"):
            require_string(record, key, owner)
        category = require_string(record, "category", owner)
        if category not in allowed_categories:
            fail(f"{owner} has invalid category: {category}")
        check_evidence_refs(record, owner, root)
    return method_ids


def validate_paper_baselines(data: dict[str, Any]) -> set[str]:
    records = require_list(data, "paper_baselines", "paper_baselines")
    baseline_ids = check_unique_ids(records, "paper baseline")
    for record in records:
        owner = f"paper baseline {record['id']}"
        for key in ("name", "paper_role", "status", "next_action"):
            require_string(record, key, owner)
        source = require_dict(record, "source", owner)
        for key in ("upstream_url", "local_tmp_path", "commit"):
            require_string(source, key, owner)
        if len(source["commit"]) != 40:
            fail(f"{owner} source commit is not pinned: {source['commit']}")
        require_list(record, "paper_baselines_to_reproduce", owner)
    return baseline_ids


def validate_paper_baseline_runs(
    data: dict[str, Any],
    baseline_ids: set[str],
    paper_evaluation_ids: set[str],
    serving_workload_ids: set[str],
) -> None:
    records = require_list(data, "paper_baseline_runs", "paper baseline runs")
    run_ids = check_unique_ids(records, "paper baseline run")
    required_runs = {
        "mpk_qwen3_native_vs_persistent",
        "vdcores_llama_decode_correctness",
        "vllm_serving_and_throughput",
        "sglang_serving_and_offline",
        "thunderkittens_tile_kernel",
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
        for artifact in record["expected_artifacts"]:
            if not artifact.startswith("tmp/"):
                fail(f"{owner} expected artifact must be under tmp/: {artifact}")

        import_target = require_dict(record, "import_target", owner)
        if (
            require_string(import_target, "viewer_file", owner)
            != "docs/nvidia-backend/benchmark-viewer/data/results.json"
        ):
            fail(f"{owner} import target must be viewer results.json")
        for key in ("result_kind", "notes"):
            require_string(import_target, key, owner)

    if not required_baseline_coverage <= covered_baselines:
        missing = sorted(required_baseline_coverage - covered_baselines)
        fail(f"paper baseline runs missing baseline coverage: {missing}")


def validate_serving_workloads(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "serving_workloads", "serving workloads")
    serving_ids = check_unique_ids(records, "serving workload")
    required_workloads = {"mpk_offline_decode", "vdcores_offline_decode"}
    if not required_workloads <= serving_ids:
        missing = sorted(required_workloads - serving_ids)
        fail(f"missing serving workloads: {missing}")

    allowed_status = {
        "policy_selected_no_results",
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
        if not (root / source["evidence"]).is_file():
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


def validate_paper_baseline_probes(
    data: dict[str, Any],
    baseline_ids: set[str],
) -> None:
    records = require_list(data, "paper_baseline_probes", "paper baseline probes")
    check_unique_ids(records, "paper baseline probe")
    allowed_status = {"not_captured", "pass", "partial", "fail"}
    allowed_kinds = {
        "path_exists",
        "py_compile",
        "python_import",
        "python_module",
    }
    covered_baselines: set[str] = set()
    for record in records:
        owner = f"paper baseline probe {record['id']}"
        for key in ("title", "latest_status", "latest_artifact_root", "next_action"):
            require_string(record, key, owner)
        if record["latest_status"] not in allowed_status:
            fail(f"{owner} has invalid latest_status: {record['latest_status']}")
        if not record["latest_artifact_root"].startswith("tmp/"):
            fail(f"{owner} latest_artifact_root must be under tmp/")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        covered_baselines.add(baseline_id)
        checks = require_list(record, "checks", owner)
        modules: set[str] = set()
        for check in checks:
            if not isinstance(check, dict):
                fail(f"{owner} check is not an object")
            kind = require_string(check, "kind", owner)
            if kind not in allowed_kinds:
                fail(f"{owner} has invalid check kind: {kind}")
            require_string(check, "why", owner)
            if kind in {"path_exists", "py_compile"}:
                require_string(check, "path", owner)
            if kind in {"python_import", "python_module"}:
                modules.add(require_string(check, "module", owner))
            if kind == "python_import" and "pythonpath" in check:
                require_string(check, "pythonpath", owner)
        if baseline_id == "thunderkittens":
            required_modules = {
                "torch",
                "pybind11",
                "numpy",
                "pandas",
                "matplotlib",
                "tqdm",
            }
            if not required_modules <= modules:
                missing = sorted(required_modules - modules)
                fail(f"{owner} missing ThunderKittens modules: {missing}")
        if baseline_id in {"mpk", "vdcores"} and "transformers" not in modules:
            fail(f"{owner} missing Transformers module probe")
    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    if not required_baselines <= covered_baselines:
        missing = sorted(required_baselines - covered_baselines)
        fail(f"paper baseline probes missing baseline coverage: {missing}")


def validate_capture_imports(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
) -> None:
    hardware = require_dict(data, "hardware", "capture imports")
    for machine, record in hardware.items():
        if not isinstance(machine, str) or not machine:
            fail("capture imports hardware machine is empty")
        if not isinstance(record, dict):
            fail(f"capture imports hardware {machine} is not an object")
        for key in ("gpu", "compute_target"):
            require_string(record, key, f"capture imports hardware {machine}")

    records = require_list(data, "capture_imports", "capture imports")
    baselines: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            fail("capture import rule is not an object")
        owner = f"capture import {record.get('baseline', '<missing>')}"
        baseline = require_string(record, "baseline", owner)
        validate_id(baseline, owner)
        if baseline in baselines:
            fail(f"duplicate capture import baseline: {baseline}")
        baselines.add(baseline)
        benchmark_id = require_string(record, "benchmark_id", owner)
        method_id = require_string(record, "method_id", owner)
        if benchmark_id not in benchmark_ids:
            fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{owner} references unknown method_id: {method_id}")
        for key in ("n", "task_count"):
            value = record.get(key)
            if not isinstance(value, int) or value <= 0:
                fail(f"{owner} has invalid {key}")
        inputs = require_dict(record, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)


def validate_results(
    data: dict[str, Any], benchmark_ids: set[str], method_ids: set[str]
) -> None:
    snapshot = require_dict(data, "snapshot", "results")
    require_string(snapshot, "commit", "results snapshot")
    if len(snapshot["commit"]) < 7:
        fail("results snapshot commit is too short")
    for key in ("full_capture", "compact_capture"):
        capture = require_dict(snapshot, key, "results snapshot")
        if not isinstance(capture.get("samples"), int) or capture["samples"] <= 0:
            fail(f"results snapshot {key} has invalid sample count")
        require_string(capture, "artifact_root", "results snapshot")

    for key in ("headline_results", "selected_rows", "result_records"):
        require_list(data, key, "results")

    for record in data["result_records"]:
        if not isinstance(record, dict):
            fail("result record is not an object")
        owner = f"result {record.get('benchmark_id', '<missing>')}"
        benchmark_id = require_string(record, "benchmark_id", owner)
        method_id = require_string(record, "method_id", owner)
        if benchmark_id not in benchmark_ids:
            fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{owner} references unknown method_id: {method_id}")
        require_string(record, "commit", owner)
        hardware = require_dict(record, "hardware", owner)
        for key in ("gpu", "machine", "compute_target", "driver", "cuda_toolkit"):
            require_string(hardware, key, owner)
        inputs = require_dict(record, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)
        statistic = require_dict(record, "statistic", owner)
        require_string(statistic, "kind", owner)
        sample_count = statistic.get("sample_count")
        if not isinstance(sample_count, int) or sample_count <= 0:
            fail(f"{owner} has invalid statistic.sample_count")
        for key in ("host_wall_ns", "device_wall_ns"):
            if not isinstance(statistic.get(key), int) or statistic[key] < 0:
                fail(f"{owner} has invalid statistic.{key}")
        raw_artifact = require_string(record, "raw_artifact", owner)
        if not raw_artifact.startswith("tmp/"):
            fail(f"{owner} raw_artifact must be under tmp/: {raw_artifact}")
        if require_string(record, "correctness", owner) not in {
            "pass",
            "fail",
            "skipped",
            "not_applicable",
        }:
            fail(f"{owner} has invalid correctness: {record['correctness']}")


def validate_paper_evaluation_matrix(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
    baseline_ids: set[str],
    results: dict[str, Any],
    root: Path,
) -> set[str]:
    records = require_list(
        data, "paper_evaluation_matrix", "paper evaluation matrix"
    )
    matrix_ids = check_unique_ids(records, "paper evaluation matrix")
    required_claims = {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    }
    if not required_claims <= matrix_ids:
        missing = sorted(required_claims - matrix_ids)
        fail(f"missing paper evaluation matrix claims: {missing}")

    result_index = {
        (
            result["benchmark_id"],
            result["method_id"],
            result["hardware"]["gpu"],
        )
        for result in results["result_records"]
    }
    baseline_coverage: set[str] = set()
    method_coverage: set[str] = set()
    hardware_coverage: set[str] = set()
    allowed_status = {
        "planned_no_results",
        "partial_current_capture",
        "ready_for_paper_claim",
    }
    required_metrics = {"correctness", "raw_artifacts"}

    for record in records:
        owner = f"paper evaluation matrix {record['id']}"
        for key in ("title", "claim", "status", "promotion_gate"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")

        workloads = require_list(record, "workload_ids", owner)
        methods = require_list(record, "method_ids", owner)
        baselines = record.get("paper_baseline_ids", [])
        if not isinstance(baselines, list):
            fail(f"{owner} paper_baseline_ids is not a list")
        hardware_targets = require_list(record, "hardware_targets", owner)
        metrics = set(require_list(record, "required_metrics", owner))
        evidence_refs = require_list(record, "current_evidence_refs", owner)
        require_list(record, "missing_evidence", owner)

        for workload_id in workloads:
            if workload_id not in benchmark_ids:
                fail(f"{owner} references unknown workload_id: {workload_id}")
        for method_id in methods:
            if method_id not in method_ids:
                fail(f"{owner} references unknown method_id: {method_id}")
            method_coverage.add(method_id)
        for baseline_id in baselines:
            if baseline_id not in baseline_ids:
                fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
            baseline_coverage.add(baseline_id)
        for hardware in hardware_targets:
            if not isinstance(hardware, str) or not hardware:
                fail(f"{owner} has invalid hardware target")
            hardware_coverage.add(hardware)
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                fail(f"{owner} has invalid required metric")
        if not required_metrics <= metrics:
            missing = sorted(required_metrics - metrics)
            fail(f"{owner} missing required metrics: {missing}")

        for ref in evidence_refs:
            if not isinstance(ref, dict):
                fail(f"{owner} current evidence ref is not an object")
            kind = require_string(ref, "kind", owner)
            if kind == "viewer_result":
                key = (
                    require_string(ref, "benchmark_id", owner),
                    require_string(ref, "method_id", owner),
                    require_string(ref, "gpu", owner),
                )
                if key not in result_index:
                    fail(f"{owner} viewer_result evidence is missing: {key}")
            elif kind in {
                "viewer_data",
                "stable_doc",
                "baseline_survey",
            }:
                path = require_string(ref, "path", owner)
                if not (root / path).is_file():
                    fail(f"{owner} evidence path missing: {path}")
            elif kind == "raw_artifact":
                path = require_string(ref, "path", owner)
                if not path.startswith("tmp/"):
                    fail(f"{owner} raw artifact evidence must be under tmp/: {path}")
            else:
                fail(f"{owner} has unknown evidence kind: {kind}")

    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    if not required_baselines <= baseline_coverage:
        missing = sorted(required_baselines - baseline_coverage)
        fail(f"paper evaluation matrix missing baseline coverage: {missing}")
    required_methods = {"pto_host_schedule", "pto_persistent_device"}
    if not required_methods <= method_coverage:
        missing = sorted(required_methods - method_coverage)
        fail(f"paper evaluation matrix missing PTO method coverage: {missing}")
    if not {"A100", "H200"} <= hardware_coverage:
        fail("paper evaluation matrix must cover A100 and H200")
    return matrix_ids


def validate_viewer_data(root: Path = ROOT) -> None:
    benchmarks = load_json(root, "benchmarks.json")
    methods = load_json(root, "methods.json")
    paper_baselines = load_json(root, "paper_baselines.json")
    paper_baseline_runs = load_json(root, "paper_baseline_runs.json")
    paper_baseline_probes = load_json(root, "paper_baseline_probes.json")
    serving_workloads = load_json(root, "serving_workloads.json")
    paper_evaluation_matrix = load_json(root, "paper_evaluation_matrix.json")
    capture_imports = load_json(root, "capture_imports.json")
    results = load_json(root, "results.json")
    benchmark_ids = validate_benchmarks(benchmarks, root)
    method_ids = validate_methods(methods, root)
    baseline_ids = validate_paper_baselines(paper_baselines)
    serving_workload_ids = validate_serving_workloads(serving_workloads, root)
    validate_paper_baseline_probes(paper_baseline_probes, baseline_ids)
    validate_capture_imports(capture_imports, benchmark_ids, method_ids)
    validate_results(results, benchmark_ids, method_ids)
    paper_evaluation_ids = validate_paper_evaluation_matrix(
        paper_evaluation_matrix,
        benchmark_ids,
        method_ids,
        baseline_ids,
        results,
        root,
    )
    validate_paper_baseline_runs(
        paper_baseline_runs,
        baseline_ids,
        paper_evaluation_ids,
        serving_workload_ids,
    )
    validate_serving_workload_run_refs(
        serving_workloads,
        {
            record["id"]
            for record in paper_baseline_runs["paper_baseline_runs"]
        },
    )


def main() -> None:
    validate_viewer_data()
    print("benchmark viewer data validation passed")


if __name__ == "__main__":
    main()
