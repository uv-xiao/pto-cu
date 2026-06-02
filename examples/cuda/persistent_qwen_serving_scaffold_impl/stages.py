"""Stage construction for the Qwen persistent serving scaffold."""

from __future__ import annotations

from typing import Any

from .common import TARGET_WORKLOAD_IDS, VIEWER_DATA, load_json


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


def core_stages(artifacts: dict[str, dict[str, Any]], flags: dict[str, Any]) -> list[dict[str, Any]]:
    lifecycle_plan = artifacts["lifecycle_plan"]
    prompt_accounting = artifacts["prompt_accounting"]
    return [
        stage(
            stage_id="serving_workload_contract",
            title="Shared Qwen serving workload contracts",
            owner="benchmark_viewer",
            required_for_full_serving=True,
            status="pass" if serving_workload_contracts() else "fail",
            evidence="evaluations/nvidia/benchmark-viewer/data/serving_workloads.json",
            next_action="Keep MPK and VDCores serving policies in viewer data.",
        ),
        stage(
            stage_id="persistent_device_task_abi",
            title="Persistent-device task ABI",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if flags["persistent_abi_ready"] else "fail",
            evidence="src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            next_action="Expose any additional tensor/scalar slots required by Qwen decode tasks.",
        ),
        stage(
            stage_id="persistent_dag_codegen",
            title="Persistent DAG task code generation",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if flags["persistent_codegen_ready"] else "fail",
            evidence="simpler_setup/cuda_callable_compiler.py",
            next_action="Add generated Qwen decode task bodies once kernels are selected.",
        ),
        stage(
            stage_id="qwen_serving_lifecycle_plan",
            title="Qwen persistent-device lifecycle plan",
            owner="pto_serving_host",
            required_for_full_serving=False,
            status="pass"
            if lifecycle_plan.get("kind") == "pto_qwen_persistent_serving_lifecycle_plan"
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
    ]


def input_stages(artifacts: dict[str, dict[str, Any]], flags: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_input = artifacts["runtime_input_binding"]
    cuda_token = artifacts["cuda_token_buffer_binding"]
    decode_args = artifacts["persistent_decode_args"]
    token_table = artifacts["token_pointer_table"]
    return [
        stage(
            stage_id="qwen_runtime_input_binding",
            title="Qwen runtime token input binding",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass" if flags["runtime_input_binding_ready"] else "missing"
                if runtime_input.get("kind") != "pto_qwen_runtime_input_binding"
                else "fail"
            ),
            evidence="examples/cuda/qwen_runtime_input_binding.py",
            next_action="Keep target-length input_ids and attention masks in sync with CUDA token-buffer binding.",
        ),
        stage(
            stage_id="qwen_cuda_token_buffer_binding",
            title="Qwen CUDA token buffer binding",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status=(
                "pass" if flags["cuda_token_buffer_status"] == "cuda_token_buffer_binding_ready"
                else "partial" if flags["cuda_token_buffer_status"] == "token_buffer_binding_plan_ready"
                else "missing" if cuda_token.get("kind") != "pto_qwen_cuda_token_buffer_binding"
                else "fail"
            ),
            evidence="examples/cuda/qwen_cuda_token_buffer_binding.py",
            next_action="Keep CUDA token-buffer ownership open through persistent decode argument binding.",
        ),
        stage(
            stage_id="qwen_persistent_decode_args",
            title="Qwen persistent decode argument binding",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status=(
                "pass" if flags["persistent_decode_args_status"] == "persistent_decode_args_ready"
                else "partial" if flags["persistent_decode_args_status"] == "persistent_decode_args_plan_ready"
                else "missing" if decode_args.get("kind") != "pto_qwen_persistent_decode_args"
                else "fail"
            ),
            evidence="examples/cuda/qwen_persistent_decode_args.py",
            next_action=(
                "Provide live token pointer tables from the decode-loop runner, "
                "then run kernels that consume those fields."
            ),
        ),
        stage(
            stage_id="qwen_token_pointer_table_owner",
            title="Qwen token pointer table owner",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status=(
                "partial" if flags["token_pointer_table_ready"] and token_table.get("mode") == "dry_run_pointer_lifecycle"
                else "pass" if flags["token_pointer_table_ready"] else "missing"
            ),
            evidence="examples/cuda/qwen_token_pointer_table.py",
            next_action=(
                "Run the token pointer table owner in cuda_live mode from "
                "the decode-loop runner and keep it open through DAG submission."
            ),
        ),
    ]


def weight_stages(artifacts: dict[str, dict[str, Any]], flags: dict[str, Any]) -> list[dict[str, Any]]:
    weight_inventory = artifacts["weight_inventory"]
    cuda_weight = artifacts["cuda_weight_binding"]
    weight_args = artifacts["persistent_weight_args"]
    materialization = artifacts["persistent_weight_materialization"]
    resident = artifacts["resident_weight_table"]
    shards = artifacts["safetensors_shards"]
    return [
        stage(
            stage_id="qwen_weight_loader",
            title="Qwen weight loader",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="partial" if weight_inventory.get("kind") == "pto_qwen_weight_inventory" else "missing",
            evidence="examples/cuda/qwen_weight_inventory.py",
            next_action=flags["weight_next_action"],
        ),
        stage(
            stage_id="qwen_cuda_weight_binding",
            title="Qwen CUDA weight binding plan",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="pass" if flags["weight_binding_ready"] else "partial"
            if cuda_weight.get("kind") == "pto_qwen_cuda_weight_binding" else "missing",
            evidence="examples/cuda/qwen_cuda_weight_binding.py",
            next_action="Run the CUDA copy probe and connect resident device pointers to persistent-device task args.",
        ),
        stage(
            stage_id="qwen_persistent_weight_args",
            title="Qwen persistent weight argument manifest",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="pass" if flags["persistent_weight_args_ready"] else "partial"
            if weight_args.get("kind") == "pto_qwen_persistent_weight_args" else "missing",
            evidence="examples/cuda/qwen_persistent_weight_args.py",
            next_action="Materialize these tensor_arg descriptors with resident device pointers in the decode-loop runner.",
        ),
        stage(
            stage_id="qwen_persistent_weight_materialization",
            title="Qwen persistent weight pointer materialization",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="pass" if materialization.get("status") == "persistent_weight_materialization_ready"
            else "partial" if flags["persistent_weight_materialization_planned"] else "missing",
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
            status="partial" if flags["resident_weight_table_ready"] and resident.get("mode") == "dry_run_pointer_lifecycle"
            else "pass" if flags["resident_weight_table_ready"] else "missing",
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
            status="pass" if flags["shards_ready"] else "partial"
            if shards.get("kind") == "pto_qwen_safetensors_shard_status" else "missing",
            evidence="examples/cuda/qwen_safetensors_fetch.py",
            next_action=flags["shard_next_action"],
        ),
    ]


def build_stages(artifacts: dict[str, dict[str, Any]], flags: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    stages.extend(core_stages(artifacts, flags))
    stages.extend(input_stages(artifacts, flags))
    stages.extend(weight_stages(artifacts, flags))
    from .runtime_stages import runtime_stages

    stages.extend(runtime_stages(artifacts, flags))
    return stages
