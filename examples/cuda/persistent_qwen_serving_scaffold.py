#!/usr/bin/env python3
"""Emit the PTO CUDA persistent-device Qwen serving lifecycle scaffold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
LIFECYCLE_PLAN = ROOT / "examples" / "cuda" / "qwen_serving_lifecycle_plan.py"
PROMPT_ACCOUNTING = ROOT / "examples" / "cuda" / "qwen_prompt_accounting.py"
RUNTIME_INPUT_BINDING = (
    ROOT / "examples" / "cuda" / "qwen_runtime_input_binding.py"
)
CUDA_TOKEN_BUFFER_BINDING = (
    ROOT / "examples" / "cuda" / "qwen_cuda_token_buffer_binding.py"
)
PERSISTENT_DECODE_ARGS = (
    ROOT / "examples" / "cuda" / "qwen_persistent_decode_args.py"
)
TOKEN_POINTER_TABLE = ROOT / "examples" / "cuda" / "qwen_token_pointer_table.py"
WEIGHT_INVENTORY = ROOT / "examples" / "cuda" / "qwen_weight_inventory.py"
SAFETENSORS_FETCH = ROOT / "examples" / "cuda" / "qwen_safetensors_fetch.py"
SAFETENSORS_METADATA = (
    ROOT / "examples" / "cuda" / "qwen_safetensors_metadata.py"
)
CUDA_WEIGHT_BINDING = ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py"
PERSISTENT_WEIGHT_ARGS = (
    ROOT / "examples" / "cuda" / "qwen_persistent_weight_args.py"
)
PERSISTENT_WEIGHT_MATERIALIZATION = (
    ROOT / "examples" / "cuda" / "qwen_persistent_weight_materialization.py"
)
RESIDENT_WEIGHT_TABLE = ROOT / "examples" / "cuda" / "qwen_resident_weight_table.py"
KV_CACHE_BINDING = ROOT / "examples" / "cuda" / "qwen_kv_cache_binding.py"
DECODE_LOOP_RUNNER = ROOT / "examples" / "cuda" / "qwen_decode_loop_runner.py"
TASK_BODIES = ROOT / "examples" / "cuda" / "qwen_persistent_task_bodies.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def text_contains(path: str, needles: list[str]) -> bool:
    full_path = ROOT / path
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def load_python_payload(path: Path, module_name: str, build_name: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    build = getattr(module, build_name)
    return build()


def load_lifecycle_plan() -> dict[str, Any]:
    return load_python_payload(
        LIFECYCLE_PLAN,
        "qwen_serving_lifecycle_plan",
        "build_lifecycle_plan",
    )


def load_prompt_accounting() -> dict[str, Any]:
    return load_python_payload(
        PROMPT_ACCOUNTING,
        "qwen_prompt_accounting",
        "build_prompt_accounting",
    )


def load_runtime_input_binding() -> dict[str, Any]:
    return load_python_payload(
        RUNTIME_INPUT_BINDING,
        "qwen_runtime_input_binding",
        "build_runtime_input_binding",
    )


def load_cuda_token_buffer_binding() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_cuda_token_buffer_binding",
        CUDA_TOKEN_BUFFER_BINDING,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_cuda_token_buffer_binding(no_cuda_probe=True)


def load_persistent_decode_args() -> dict[str, Any]:
    return load_python_payload(
        PERSISTENT_DECODE_ARGS,
        "qwen_persistent_decode_args",
        "build_decode_arg_manifest",
    )


def load_token_pointer_table() -> dict[str, Any]:
    return load_python_payload(
        TOKEN_POINTER_TABLE,
        "qwen_token_pointer_table",
        "build_token_pointer_table_lifecycle",
    )


def load_weight_inventory() -> dict[str, Any]:
    return load_python_payload(
        WEIGHT_INVENTORY,
        "qwen_weight_inventory",
        "build_weight_inventory",
    )


def load_safetensors_shards() -> dict[str, Any]:
    return load_python_payload(
        SAFETENSORS_FETCH,
        "qwen_safetensors_fetch",
        "build_shard_status",
    )


def load_safetensors_metadata() -> dict[str, Any]:
    return load_python_payload(
        SAFETENSORS_METADATA,
        "qwen_safetensors_metadata",
        "build_metadata_probe",
    )


def load_cuda_weight_binding() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_cuda_weight_binding",
        CUDA_WEIGHT_BINDING,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_weight_binding(no_cuda_probe=True)


def load_persistent_weight_args() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_weight_args",
        PERSISTENT_WEIGHT_ARGS,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_weight_arg_manifest()


def load_persistent_weight_materialization() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_weight_materialization",
        PERSISTENT_WEIGHT_MATERIALIZATION,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_materialization_manifest()


def load_resident_weight_table() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_resident_weight_table",
        RESIDENT_WEIGHT_TABLE,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_resident_table_lifecycle()


def load_kv_cache_binding() -> dict[str, Any]:
    return load_python_payload(
        KV_CACHE_BINDING,
        "qwen_kv_cache_binding",
        "build_kv_cache_lifecycle",
    )


def load_decode_loop_runner() -> dict[str, Any]:
    return load_python_payload(
        DECODE_LOOP_RUNNER,
        "qwen_decode_loop_runner",
        "build_decode_loop_runner",
    )


def load_task_bodies() -> dict[str, Any]:
    return load_python_payload(
        TASK_BODIES,
        "qwen_persistent_task_bodies",
        "build_task_body_manifest",
    )


def serving_workload_contracts() -> list[dict[str, Any]]:
    payload = load_json(VIEWER_DATA / "serving_workloads.json")
    workloads = []
    for item in payload.get("serving_workloads", []):
        if item.get("id") not in TARGET_WORKLOAD_IDS:
            continue
        model_policy = item.get("model_policy", {})
        prompt_policy = item.get("prompt_policy", {})
        decode_policy = item.get("decode_policy", {})
        workloads.append(
            {
                "id": item.get("id"),
                "primary_model": model_policy.get("primary_model"),
                "target_prompt_tokens": prompt_policy.get("target_prompt_tokens"),
                "decode_tokens": decode_policy.get("decode_tokens"),
                "batch_sizes": decode_policy.get("batch_sizes", []),
            }
        )
    return workloads


def stage(
    *,
    stage_id: str,
    title: str,
    owner: str,
    required_for_full_serving: bool,
    status: str,
    evidence: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "title": title,
        "owner": owner,
        "required_for_full_serving": required_for_full_serving,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def build_scaffold() -> dict[str, Any]:
    lifecycle_plan = load_lifecycle_plan()
    prompt_accounting = load_prompt_accounting()
    runtime_input_binding = load_runtime_input_binding()
    cuda_token_buffer_binding = load_cuda_token_buffer_binding()
    persistent_decode_args = load_persistent_decode_args()
    token_pointer_table = load_token_pointer_table()
    weight_inventory = load_weight_inventory()
    safetensors_shards = load_safetensors_shards()
    safetensors_metadata = load_safetensors_metadata()
    cuda_weight_binding = load_cuda_weight_binding()
    persistent_weight_args = load_persistent_weight_args()
    persistent_weight_materialization = load_persistent_weight_materialization()
    resident_weight_table = load_resident_weight_table()
    kv_cache_binding = load_kv_cache_binding()
    decode_loop_runner = load_decode_loop_runner()
    task_bodies = load_task_bodies()
    shards_ready = (
        safetensors_shards.get("status") == "ready_for_metadata_probe"
    )
    metadata_validated = (
        safetensors_metadata.get("status") == "metadata_validated"
    )
    weight_binding_ready = (
        cuda_weight_binding.get("status") == "binding_plan_ready"
    )
    persistent_weight_args_ready = (
        persistent_weight_args.get("status") == "persistent_weight_args_ready"
    )
    persistent_weight_materialization_planned = (
        persistent_weight_materialization.get("kind")
        == "pto_qwen_persistent_weight_materialization"
        and persistent_weight_materialization.get("status")
        in {
            "persistent_weight_materialization_plan_ready",
            "persistent_weight_materialization_ready",
        }
    )
    resident_weight_table_ready = (
        resident_weight_table.get("status")
        == "resident_weight_table_lifecycle_ready"
    )
    kv_cache_binding_ready = (
        kv_cache_binding.get("status") == "kv_cache_lifecycle_ready"
    )
    decode_loop_runner_ready = (
        decode_loop_runner.get("status") == "decode_loop_runner_plan_ready"
    )
    task_bodies_ready = (
        task_bodies.get("status") == "generated_task_bodies_ready"
    )
    runtime_input_binding_ready = (
        runtime_input_binding.get("status") == "runtime_input_binding_plan_ready"
    )
    cuda_token_buffer_status = cuda_token_buffer_binding.get("status")
    persistent_decode_args_status = persistent_decode_args.get("status")
    token_pointer_table_ready = (
        token_pointer_table.get("status") == "token_pointer_table_lifecycle_ready"
    )
    weight_next_action = (
        "Connect the resident weight table owner to the decode-loop runner and "
        "replace dry-run pointers with cuda_live table ownership."
        if resident_weight_table_ready
        else (
            "Create the live decode-loop resident pointer table, then pass it to "
            "the persistent weight materializer before launching Qwen tasks."
        )
        if persistent_weight_materialization_planned
        else (
            "Materialize persistent task descriptors with resident weight "
            "pointers during the decode-loop runner."
        )
        if persistent_weight_args_ready
        else (
            "Move the planned CUDA weight bindings into real persistent task "
            "arguments and full device residency."
        )
        if weight_binding_ready
        else (
            "Bind validated Qwen safetensors tensors to CUDA buffers and "
            "persistent-device task args."
        )
        if metadata_validated
        else (
            "Run the safetensors metadata probe against the placed Qwen "
            "shards, then bind validated tensors to CUDA buffers."
            if shards_ready
            else (
                "Keep the safetensors index and expected shape/dtype contract "
                "in sync with the shard placement and metadata probes."
            )
        )
    )
    shard_next_action = (
        "Keep the local Qwen safetensors shards under tmp/sources and rerun "
        "the metadata probe when shard files change."
        if shards_ready
        else (
            "Place or download the real Qwen safetensors shards under "
            "tmp/sources/qwen3-8b-safetensors, then rerun the metadata probe."
        )
    )
    persistent_abi_ready = text_contains(
        "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
        ["PtoCudaPersistentDagTask", "tensor_args", "scalar_args"],
    )
    persistent_codegen_ready = text_contains(
        "simpler_setup/cuda_callable_compiler.py",
        ["render_persistent_dag_source", "CudaPersistentTaskBodyFunction"],
    )
    stages = [
        stage(
            stage_id="serving_workload_contract",
            title="Shared Qwen serving workload contracts",
            owner="benchmark_viewer",
            required_for_full_serving=True,
            status="pass" if serving_workload_contracts() else "fail",
            evidence=(
                "docs/nvidia-backend/benchmark-viewer/data/"
                "serving_workloads.json"
            ),
            next_action="Keep MPK and VDCores serving policies in viewer data.",
        ),
        stage(
            stage_id="persistent_device_task_abi",
            title="Persistent-device task ABI",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if persistent_abi_ready else "fail",
            evidence="src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            next_action="Expose any additional tensor/scalar slots required by Qwen decode tasks.",
        ),
        stage(
            stage_id="persistent_dag_codegen",
            title="Persistent DAG task code generation",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if persistent_codegen_ready else "fail",
            evidence="simpler_setup/cuda_callable_compiler.py",
            next_action="Add generated Qwen decode task bodies once kernels are selected.",
        ),
        stage(
            stage_id="qwen_serving_lifecycle_plan",
            title="Qwen persistent-device lifecycle plan",
            owner="pto_serving_host",
            required_for_full_serving=False,
            status="pass"
            if lifecycle_plan.get("kind")
            == "pto_qwen_persistent_serving_lifecycle_plan"
            else "fail",
            evidence="examples/cuda/qwen_serving_lifecycle_plan.py",
            next_action=(
                "Keep the model-shape, KV-cache, and task-mapping plan in "
                "sync with the runtime implementation."
            ),
        ),
        stage(
            stage_id="qwen_tokenizer",
            title="Qwen tokenizer and prompt accounting",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="partial"
            if prompt_accounting.get("kind") == "pto_qwen_prompt_accounting"
            else "missing",
            evidence="examples/cuda/qwen_prompt_accounting.py",
            next_action=(
                "Keep tokenizer accounting synchronized with runtime input "
                "binding and target prompt alignment policy."
            ),
        ),
        stage(
            stage_id="qwen_runtime_input_binding",
            title="Qwen runtime token input binding",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if runtime_input_binding_ready
                else "missing"
                if runtime_input_binding.get("kind") != "pto_qwen_runtime_input_binding"
                else "fail"
            ),
            evidence="examples/cuda/qwen_runtime_input_binding.py",
            next_action=(
                "Keep target-length input_ids and attention masks in sync "
                "with CUDA token-buffer binding."
            ),
        ),
        stage(
            stage_id="qwen_cuda_token_buffer_binding",
            title="Qwen CUDA token buffer binding",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if cuda_token_buffer_status == "cuda_token_buffer_binding_ready"
                else "partial"
                if cuda_token_buffer_status == "token_buffer_binding_plan_ready"
                else "missing"
                if cuda_token_buffer_binding.get("kind")
                != "pto_qwen_cuda_token_buffer_binding"
                else "fail"
            ),
            evidence="examples/cuda/qwen_cuda_token_buffer_binding.py",
            next_action=(
                "Keep CUDA token-buffer ownership open through persistent "
                "decode argument binding."
            ),
        ),
        stage(
            stage_id="qwen_persistent_decode_args",
            title="Qwen persistent decode argument binding",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status=(
                "pass"
                if persistent_decode_args_status == "persistent_decode_args_ready"
                else "partial"
                if persistent_decode_args_status
                == "persistent_decode_args_plan_ready"
                else "missing"
                if persistent_decode_args.get("kind")
                != "pto_qwen_persistent_decode_args"
                else "fail"
            ),
            evidence="examples/cuda/qwen_persistent_decode_args.py",
            next_action=(
                "Provide live token pointer tables from the decode-loop "
                "runner, then run kernels that consume those fields."
            ),
        ),
        stage(
            stage_id="qwen_token_pointer_table_owner",
            title="Qwen token pointer table owner",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status=(
                "partial"
                if token_pointer_table_ready
                and token_pointer_table.get("mode") == "dry_run_pointer_lifecycle"
                else "pass"
                if token_pointer_table_ready
                else "missing"
            ),
            evidence="examples/cuda/qwen_token_pointer_table.py",
            next_action=(
                "Run the token pointer table owner in cuda_live mode from "
                "the decode-loop runner and keep it open through DAG submission."
            ),
        ),
        stage(
            stage_id="qwen_weight_loader",
            title="Qwen weight loader",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="partial"
            if weight_inventory.get("kind") == "pto_qwen_weight_inventory"
            else "missing",
            evidence="examples/cuda/qwen_weight_inventory.py",
            next_action=weight_next_action,
        ),
        stage(
            stage_id="qwen_cuda_weight_binding",
            title="Qwen CUDA weight binding plan",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if weight_binding_ready
                else "partial"
                if cuda_weight_binding.get("kind")
                == "pto_qwen_cuda_weight_binding"
                else "missing"
            ),
            evidence="examples/cuda/qwen_cuda_weight_binding.py",
            next_action=(
                "Run the CUDA copy probe and connect resident device "
                "pointers to persistent-device task args."
            ),
        ),
        stage(
            stage_id="qwen_persistent_weight_args",
            title="Qwen persistent weight argument manifest",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if persistent_weight_args_ready
                else "partial"
                if persistent_weight_args.get("kind")
                == "pto_qwen_persistent_weight_args"
                else "missing"
            ),
            evidence="examples/cuda/qwen_persistent_weight_args.py",
            next_action=(
                "Materialize these tensor_arg descriptors with resident "
                "device pointers in the decode-loop runner."
            ),
        ),
        stage(
            stage_id="qwen_persistent_weight_materialization",
            title="Qwen persistent weight pointer materialization",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if persistent_weight_materialization.get("status")
                == "persistent_weight_materialization_ready"
                else "partial"
                if persistent_weight_materialization_planned
                else "missing"
            ),
            evidence="examples/cuda/qwen_persistent_weight_materialization.py",
            next_action=(
                "Make the decode-loop runner own a live resident_weight_ptrs "
                "table and call the materializer before DAG submission."
            ),
        ),
        stage(
            stage_id="qwen_resident_weight_table_owner",
            title="Qwen resident weight pointer table owner",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "partial"
                if resident_weight_table_ready
                and resident_weight_table.get("mode") == "dry_run_pointer_lifecycle"
                else "pass"
                if resident_weight_table_ready
                else "missing"
            ),
            evidence="examples/cuda/qwen_resident_weight_table.py",
            next_action=(
                "Run the resident table owner in cuda_live mode from the "
                "decode-loop runner and keep it open through DAG submission."
            ),
        ),
        stage(
            stage_id="qwen_safetensors_shards",
            title="Qwen safetensors shard placement",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass"
                if safetensors_shards.get("status") == "ready_for_metadata_probe"
                else "partial"
                if safetensors_shards.get("kind")
                == "pto_qwen_safetensors_shard_status"
                else "missing"
            ),
            evidence="examples/cuda/qwen_safetensors_fetch.py",
            next_action=shard_next_action,
        ),
        stage(
            stage_id="kv_cache_lifecycle",
            title="KV-cache allocation and token-position lifecycle",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status=(
                "partial"
                if kv_cache_binding_ready
                and kv_cache_binding.get("mode") == "dry_run_pointer_lifecycle"
                else "pass"
                if kv_cache_binding_ready
                else "missing"
                if not lifecycle_plan.get("workload_plans")
                else "partial"
            ),
            evidence="examples/cuda/qwen_kv_cache_binding.py",
            next_action=(
                "Run the KV-cache owner in cuda_live mode from the "
                "decode-loop runner and consume c/d from Qwen attention kernels."
            ),
        ),
        stage(
            stage_id="decode_loop_runner",
            title="Decode loop runner",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="partial" if decode_loop_runner_ready else "missing",
            evidence="examples/cuda/qwen_decode_loop_runner.py",
            next_action=(
                "Replace the dry-run submission plan with cuda_live resource "
                "owners and numerically correct Qwen kernels."
            ),
        ),
        stage(
            stage_id="qwen_persistent_task_bodies",
            title="Qwen persistent task body source generation",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="partial" if task_bodies_ready else "missing",
            evidence="examples/cuda/qwen_persistent_task_bodies.py",
            next_action=(
                "Replace review-oriented task bodies with numerically correct "
                "Qwen kernels, then validate them in cuda_live decode-loop "
                "execution."
            ),
        ),
        stage(
            stage_id="viewer_result_import",
            title="Full-serving viewer result import",
            owner="paper_evaluation",
            required_for_full_serving=True,
            status="missing",
            evidence="docs/nvidia-backend/benchmark-viewer/data/results.json",
            next_action=(
                "Import Qwen/Qwen3-8B PTO rows with serving_coverage="
                "full_serving or full_serving_latency_caveat."
            ),
        ),
    ]
    missing = [
        item
        for item in stages
        if item["required_for_full_serving"] and item["status"] != "pass"
    ]
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_serving_scaffold",
        "status": "partial" if missing else "pass",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "runtime": "cuda/persistent_device",
        "serving_workloads": serving_workload_contracts(),
        "lifecycle_plan": lifecycle_plan,
        "prompt_accounting": prompt_accounting,
        "runtime_input_binding": runtime_input_binding,
        "cuda_token_buffer_binding": cuda_token_buffer_binding,
        "persistent_decode_args": persistent_decode_args,
        "token_pointer_table": token_pointer_table,
        "weight_inventory": weight_inventory,
        "safetensors_shards": safetensors_shards,
        "safetensors_metadata": safetensors_metadata,
        "cuda_weight_binding": cuda_weight_binding,
        "persistent_weight_args": persistent_weight_args,
        "persistent_weight_materialization": persistent_weight_materialization,
        "resident_weight_table": resident_weight_table,
        "kv_cache_binding": kv_cache_binding,
        "decode_loop_runner": decode_loop_runner,
        "persistent_task_bodies": task_bodies,
        "stages": stages,
        "missing_stage_ids": [item["id"] for item in missing],
        "next_action": (
            "Implement the missing PTO serving host/runtime stages, then "
            "import Qwen/Qwen3-8B full-serving rows."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_scaffold()
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
