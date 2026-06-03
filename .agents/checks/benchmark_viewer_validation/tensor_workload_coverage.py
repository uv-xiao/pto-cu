from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import check_evidence_refs


def _result_index(results: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    return {
        (
            record["benchmark_id"],
            record["method_id"],
            record["hardware"]["gpu"],
            record["inputs"]["shape"],
        )
        for record in results["result_records"]
    }


def _check_result_refs(
    refs: list[Any],
    owner: str,
    result_index: set[tuple[str, str, str, str]],
) -> None:
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} result ref is not an object")
        key = (
            require_string(ref, "benchmark_id", owner),
            require_string(ref, "method_id", owner),
            require_string(ref, "gpu", owner),
        )
        shape_contains = require_string(ref, "shape_contains", owner)
        if not any(
            result[:3] == key and shape_contains in result[3]
            for result in result_index
        ):
            fail(f"{owner} result ref is missing: {(*key, shape_contains)}")


def validate_tensor_workload_coverage(
    data: dict[str, Any],
    results: dict[str, Any],
    method_ids: set[str],
    root: Path,
) -> None:
    metadata = require_dict(data, "metadata", "tensor workload coverage")
    for key in ("title", "status", "summary"):
        require_string(metadata, key, "tensor workload coverage metadata")
    groups = require_list(data, "coverage_groups", "tensor workload coverage")
    group_ids = check_unique_ids(groups, "tensor workload coverage group")
    required = {
        "pto_tensor_descriptor_paths",
        "library_and_generated_baselines",
        "tuned_pto_tensor_body_gap",
    }
    missing = required - group_ids
    if missing:
        fail(f"tensor workload coverage missing groups: {sorted(missing)}")
    validate_model_shape_targets(data, method_ids, root)
    results_seen = _result_index(results)
    for group in groups:
        owner = f"tensor workload coverage group {group['id']}"
        for key in ("title", "status", "summary"):
            require_string(group, key, owner)
        covered = require_list(group, "covered_cases", owner)
        if len(covered) < 5:
            fail(f"{owner} must list at least five covered cases")
        result_refs = group.get("result_refs", [])
        if group["status"] != "open" and not result_refs:
            fail(f"{owner} must include result refs")
        if not isinstance(result_refs, list):
            fail(f"{owner} result_refs is not a list")
        _check_result_refs(result_refs, owner, results_seen)
        require_list(group, "open_work", owner)
        check_evidence_refs(group, owner, root)


def validate_model_shape_targets(
    data: dict[str, Any],
    method_ids: set[str],
    root: Path,
) -> None:
    targets = require_list(
        data,
        "model_shape_targets",
        "tensor workload coverage",
    )
    if len(targets) < 2:
        fail("tensor workload coverage must include at least two model shape targets")
    required_methods = {
        "pto_persistent_device",
        "cublas_sgemm_graph",
        "cutlass",
        "triton",
        "thunderkittens",
    }
    import_smoke_count = 0
    for target in targets:
        owner = f"tensor workload model shape target {target.get('id')}"
        validate_id(require_string(target, "id", owner), owner)
        for key in ("title", "model_mapping", "run_command"):
            require_string(target, key, owner)
        status = require_string(target, "status", owner)
        if status not in {"planned_shape_target", "local_import_smoke"}:
            fail(f"{owner} has invalid status: {status}")
        tile = require_dict(target, "tensor_tile", owner)
        rows = require_positive_int(tile, "rows", owner)
        cols = require_positive_int(tile, "cols", owner)
        inner = require_positive_int(tile, "inner", owner)
        if rows % 16 != 0 or cols % 16 != 0 or inner % 8 != 0:
            fail(f"{owner} is not compatible with WMMA tensor-core constraints")
        command = target["run_command"]
        for flag, value in (
            ("--tensor-rows", rows),
            ("--tensor-cols", cols),
            ("--tensor-inner", inner),
        ):
            if f"{flag} {value}" not in command:
                fail(f"{owner} command missing {flag} {value}")
        methods = set(require_list(target, "required_methods", owner))
        if methods != required_methods:
            fail(f"{owner} required methods mismatch: {sorted(methods)}")
        missing_methods = methods - method_ids
        if missing_methods:
            fail(
                f"{owner} references unknown viewer methods: "
                f"{sorted(missing_methods)}"
            )
        has_import_smoke = validate_import_smoke(target, owner, methods, root)
        if status == "local_import_smoke" and not has_import_smoke:
            fail(f"{owner} local_import_smoke status lacks import_smoke")
        import_smoke_count += int(has_import_smoke)
        validate_throughput_capture(target, owner, methods, root)
        validate_generated_kernel_capture(target, owner, methods, root)
        check_evidence_refs(target, owner, root)
    if import_smoke_count < 2:
        fail("tensor workload coverage needs two model-shape import smokes")


def validate_import_smoke(
    target: dict[str, Any],
    owner: str,
    required_methods: set[str],
    root: Path,
) -> bool:
    smoke = target.get("import_smoke")
    if smoke is None:
        return False
    if not isinstance(smoke, dict):
        fail(f"{owner} import_smoke is not an object")
    status = require_string(smoke, "status", owner)
    if status not in {"pass", "partial", "fail"}:
        fail(f"{owner} import_smoke has invalid status: {status}")
    artifact_root = require_string(smoke, "artifact_root", owner)
    require_current_artifact_path(root, artifact_root, owner)
    hardware = require_dict(smoke, "hardware", owner)
    gpu = require_string(hardware, "gpu", owner)
    compute_target = require_string(hardware, "compute_target", owner)
    exported_records_path = require_string(smoke, "exported_records_path", owner)
    records_path = root / exported_records_path
    if not records_path.is_file():
        fail(f"{owner} import_smoke records path missing: {exported_records_path}")
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{owner} import_smoke records JSON is invalid: {exc}")
    if not isinstance(records, list) or not records:
        fail(f"{owner} import_smoke records are empty")
    methods = set(require_list(smoke, "methods", owner))
    if not methods <= required_methods:
        fail(f"{owner} import_smoke methods are not required methods")
    commands = require_list(smoke, "commands", owner)
    check_import_smoke_commands(commands, methods, target, owner)
    record_methods = {record.get("method_id") for record in records}
    if not methods <= record_methods:
        fail(f"{owner} import_smoke records missing methods: {sorted(methods)}")
    sample_count = require_positive_int(smoke, "sample_count", owner)
    shape = (
        f"{target['tensor_tile']['rows']}x"
        f"{target['tensor_tile']['cols']}x"
        f"{target['tensor_tile']['inner']}"
    )
    for record in records:
        method = record.get("method_id")
        if method not in methods:
            continue
        if record.get("benchmark_id") != "tensor_core_tile":
            fail(f"{owner} import_smoke record has wrong benchmark_id")
        hardware_record = record.get("hardware", {})
        if (
            hardware_record.get("gpu") != gpu
            or hardware_record.get("compute_target") != compute_target
        ):
            fail(f"{owner} import_smoke record has wrong hardware")
        if shape not in record.get("inputs", {}).get("shape", ""):
            fail(f"{owner} import_smoke record has wrong tensor shape")
        if record.get("raw_artifact") != artifact_root:
            fail(f"{owner} import_smoke record has wrong raw_artifact")
        if status == "pass" and record.get("correctness") != "pass":
            fail(f"{owner} import_smoke record is not correctness pass")
        statistic = record.get("statistic", {})
        if statistic.get("sample_count") != sample_count:
            fail(f"{owner} import_smoke sample count mismatch")
    return True


def validate_throughput_capture(
    target: dict[str, Any],
    owner: str,
    required_methods: set[str],
    root: Path,
) -> None:
    capture = target.get("throughput_capture")
    if capture is None:
        return
    if not isinstance(capture, dict):
        fail(f"{owner} throughput_capture is not an object")
    status = require_string(capture, "status", owner)
    if status not in {"a100_multi_repeat", "a100_h200_multi_repeat"}:
        fail(f"{owner} throughput_capture has invalid status: {status}")
    artifact_root = require_string(capture, "artifact_root", owner)
    require_current_artifact_path(root, artifact_root, owner)
    hardware = require_dict(capture, "hardware", owner)
    gpu = require_string(hardware, "gpu", owner)
    compute_target = require_string(hardware, "compute_target", owner)
    if gpu not in {"A100", "H200"}:
        fail(f"{owner} throughput_capture has unsupported gpu: {gpu}")
    exported_records_path = require_string(
        capture,
        "exported_records_path",
        owner,
    )
    if not exported_records_path.startswith(artifact_root):
        fail(f"{owner} throughput_capture records must live under artifact_root")
    records_path = root / exported_records_path
    if not records_path.is_file():
        fail(
            f"{owner} throughput_capture records path missing: "
            f"{exported_records_path}"
        )
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{owner} throughput_capture records JSON is invalid: {exc}")
    if not isinstance(records, list) or not records:
        fail(f"{owner} throughput_capture records are empty")
    methods = set(require_list(capture, "methods", owner))
    if not methods <= required_methods:
        fail(f"{owner} throughput_capture methods are not required methods")
    if "pto_persistent_device" not in methods:
        fail(f"{owner} throughput_capture must include PTO rows")
    sample_count = require_positive_int(capture, "sample_count", owner)
    if sample_count < 3:
        fail(f"{owner} throughput_capture must have at least three samples")
    commands = require_list(capture, "commands", owner)
    check_import_smoke_commands(commands, methods, target, owner)
    require_string(capture, "remaining_scope", owner)

    shape = (
        f"{target['tensor_tile']['rows']}x"
        f"{target['tensor_tile']['cols']}x"
        f"{target['tensor_tile']['inner']}"
    )
    record_methods = {record.get("method_id") for record in records}
    if not methods <= record_methods:
        fail(f"{owner} throughput_capture records missing methods: {sorted(methods)}")
    for record in records:
        method = record.get("method_id")
        if method not in methods:
            continue
        if record.get("benchmark_id") != "tensor_core_tile":
            fail(f"{owner} throughput_capture record has wrong benchmark_id")
        hardware_record = record.get("hardware", {})
        if (
            hardware_record.get("gpu") != gpu
            or hardware_record.get("compute_target") != compute_target
        ):
            fail(f"{owner} throughput_capture record has wrong hardware")
        if shape not in record.get("inputs", {}).get("shape", ""):
            fail(f"{owner} throughput_capture record has wrong tensor shape")
        if record.get("raw_artifact") != artifact_root:
            fail(f"{owner} throughput_capture record has wrong raw_artifact")
        if record.get("correctness") != "pass":
            fail(f"{owner} throughput_capture record is not correctness pass")
        statistic = record.get("statistic", {})
        if statistic.get("sample_count") != sample_count:
            fail(f"{owner} throughput_capture sample count mismatch")


def validate_generated_kernel_capture(
    target: dict[str, Any],
    owner: str,
    required_methods: set[str],
    root: Path,
) -> None:
    capture = target.get("generated_kernel_capture")
    if capture is None:
        return
    if not isinstance(capture, dict):
        fail(f"{owner} generated_kernel_capture is not an object")
    status = require_string(capture, "status", owner)
    if status not in {"a100_multi_repeat", "a100_h200_multi_repeat"}:
        fail(f"{owner} generated_kernel_capture has invalid status: {status}")
    artifact_roots = require_list(capture, "artifact_roots", owner)
    exported_paths = require_list(capture, "exported_records_paths", owner)
    if len(artifact_roots) != len(exported_paths):
        fail(f"{owner} generated_kernel_capture path counts mismatch")
    for artifact_root in artifact_roots:
        if not isinstance(artifact_root, str):
            fail(f"{owner} generated_kernel_capture artifact root is not a string")
        require_current_artifact_path(root, artifact_root, owner)
    hardware = require_dict(capture, "hardware", owner)
    gpu = require_string(hardware, "gpu", owner)
    compute_target = require_string(hardware, "compute_target", owner)
    methods = set(require_list(capture, "methods", owner))
    if methods != {"triton", "cutlass"}:
        fail(f"{owner} generated_kernel_capture methods mismatch: {sorted(methods)}")
    if not methods <= required_methods:
        fail(f"{owner} generated_kernel_capture methods are not required methods")
    sample_count = require_positive_int(capture, "sample_count", owner)
    if sample_count < 3:
        fail(f"{owner} generated_kernel_capture must have at least three samples")
    commands = require_list(capture, "commands", owner)
    check_import_smoke_commands(
        commands,
        methods,
        target,
        owner,
        require_cuda_viewer_export=False,
        command_owner="generated_kernel_capture",
    )
    require_string(capture, "remaining_scope", owner)

    shape = (
        f"{target['tensor_tile']['rows']}x"
        f"{target['tensor_tile']['cols']}x"
        f"{target['tensor_tile']['inner']}"
    )
    seen_methods: set[str] = set()
    for exported_path in exported_paths:
        if not isinstance(exported_path, str):
            fail(f"{owner} generated_kernel_capture records path is not a string")
        if not any(exported_path.startswith(path) for path in artifact_roots):
            fail(f"{owner} generated_kernel_capture records path is outside roots")
        records_path = root / exported_path
        if not records_path.is_file():
            fail(
                f"{owner} generated_kernel_capture records path missing: "
                f"{exported_path}"
            )
        try:
            records = json.loads(records_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{owner} generated_kernel_capture records JSON is invalid: {exc}")
        if not isinstance(records, list) or not records:
            fail(f"{owner} generated_kernel_capture records are empty")
        for record in records:
            method = record.get("method_id")
            if method not in methods:
                fail(f"{owner} generated_kernel_capture record method mismatch")
            seen_methods.add(method)
            if record.get("benchmark_id") != "tensor_core_tile":
                fail(f"{owner} generated_kernel_capture record has wrong benchmark_id")
            hardware_record = record.get("hardware", {})
            if (
                hardware_record.get("gpu") != gpu
                or hardware_record.get("compute_target") != compute_target
            ):
                fail(f"{owner} generated_kernel_capture record has wrong hardware")
            if shape not in record.get("inputs", {}).get("shape", ""):
                fail(f"{owner} generated_kernel_capture record has wrong tensor shape")
            if not exported_path.startswith(str(record.get("raw_artifact", ""))):
                fail(f"{owner} generated_kernel_capture raw_artifact mismatch")
            if record.get("correctness") != "pass":
                fail(f"{owner} generated_kernel_capture record is not correctness pass")
            statistic = record.get("statistic", {})
            if statistic.get("sample_count") != sample_count:
                fail(f"{owner} generated_kernel_capture sample count mismatch")
    if seen_methods != methods:
        missing = sorted(methods - seen_methods)
        fail(f"{owner} generated_kernel_capture missing methods: {missing}")


def check_import_smoke_commands(
    commands: list[Any],
    methods: set[str],
    target: dict[str, Any],
    owner: str,
    require_cuda_viewer_export: bool = True,
    command_owner: str = "import_smoke",
) -> None:
    command_text = "\n".join(
        command for command in commands if isinstance(command, str)
    )
    if len(command_text.splitlines()) != len(commands):
        fail(f"{owner} {command_owner} commands must be strings")
    required_baselines = {
        "pto_persistent_device": "pto_persistent_dag_graph_tensor_core",
        "cublas_sgemm_graph": "cublas_sgemm_graph",
        "triton": "triton_tensor_tile_capture.py",
        "cutlass": "cutlass_tensor_tile_capture.py",
    }
    for method in methods:
        baseline = required_baselines.get(method)
        if baseline is None:
            continue
        if method in {"triton", "cutlass"}:
            marker = baseline
        else:
            marker = f"--single-baseline {baseline}"
        if marker not in command_text:
            fail(f"{owner} {command_owner} command missing {baseline}")
    tile = target["tensor_tile"]
    for flags, key in (
        (("--tensor-rows", "--rows"), "rows"),
        (("--tensor-cols", "--cols"), "cols"),
        (("--tensor-inner", "--inner"), "inner"),
    ):
        if not any(f"{flag} {tile[key]}" in command_text for flag in flags):
            fail(f"{owner} {command_owner} command missing {flags[0]} {tile[key]}")
    if require_cuda_viewer_export and "cuda_viewer_export.py" not in command_text:
        fail(f"{owner} {command_owner} command missing viewer export")
    if not require_cuda_viewer_export and "--viewer-output" not in command_text:
        fail(f"{owner} {command_owner} command missing direct viewer output")


def require_positive_int(record: dict[str, Any], key: str, owner: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or value <= 0:
        fail(f"{owner} has invalid positive integer {key}")
    return value
