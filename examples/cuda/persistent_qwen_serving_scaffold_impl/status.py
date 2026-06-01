"""Derived readiness flags and next actions for scaffold stages."""

from __future__ import annotations

from typing import Any

from .common import text_contains


def readiness(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safetensors_shards = artifacts["safetensors_shards"]
    safetensors_metadata = artifacts["safetensors_metadata"]
    cuda_weight_binding = artifacts["cuda_weight_binding"]
    persistent_weight_args = artifacts["persistent_weight_args"]
    materialization = artifacts["persistent_weight_materialization"]
    resident_weight_table = artifacts["resident_weight_table"]
    kv_cache_binding = artifacts["kv_cache_binding"]
    decode_loop_runner = artifacts["decode_loop_runner"]
    task_bodies = artifacts["task_bodies"]
    runtime_input_binding = artifacts["runtime_input_binding"]
    cuda_token_buffer_binding = artifacts["cuda_token_buffer_binding"]
    persistent_decode_args = artifacts["persistent_decode_args"]
    token_pointer_table = artifacts["token_pointer_table"]

    flags = {
        "shards_ready": safetensors_shards.get("status") == "ready_for_metadata_probe",
        "metadata_validated": safetensors_metadata.get("status") == "metadata_validated",
        "weight_binding_ready": cuda_weight_binding.get("status") == "binding_plan_ready",
        "persistent_weight_args_ready": (
            persistent_weight_args.get("status") == "persistent_weight_args_ready"
        ),
        "persistent_weight_materialization_planned": (
            materialization.get("kind") == "pto_qwen_persistent_weight_materialization"
            and materialization.get("status")
            in {
                "persistent_weight_materialization_plan_ready",
                "persistent_weight_materialization_ready",
            }
        ),
        "resident_weight_table_ready": (
            resident_weight_table.get("status") == "resident_weight_table_lifecycle_ready"
        ),
        "kv_cache_binding_ready": kv_cache_binding.get("status") == "kv_cache_lifecycle_ready",
        "decode_loop_runner_ready": (
            decode_loop_runner.get("status") == "decode_loop_runner_plan_ready"
        ),
        "task_bodies_ready": task_bodies.get("status") == "generated_task_bodies_ready",
        "runtime_input_binding_ready": (
            runtime_input_binding.get("status") == "runtime_input_binding_plan_ready"
        ),
        "cuda_token_buffer_status": cuda_token_buffer_binding.get("status"),
        "persistent_decode_args_status": persistent_decode_args.get("status"),
        "token_pointer_table_ready": (
            token_pointer_table.get("status") == "token_pointer_table_lifecycle_ready"
        ),
        "persistent_abi_ready": text_contains(
            "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            ["PtoCudaPersistentDagTask", "tensor_args", "scalar_args"],
        ),
        "persistent_codegen_ready": text_contains(
            "simpler_setup/cuda_callable_compiler.py",
            ["render_persistent_dag_source", "CudaPersistentTaskBodyFunction"],
        ),
    }
    flags["weight_next_action"] = weight_next_action(flags)
    flags["shard_next_action"] = shard_next_action(flags["shards_ready"])
    return flags


def weight_next_action(flags: dict[str, Any]) -> str:
    if flags["resident_weight_table_ready"]:
        return (
            "Connect the resident weight table owner to the decode-loop runner and "
            "replace dry-run pointers with cuda_live table ownership."
        )
    if flags["persistent_weight_materialization_planned"]:
        return (
            "Create the live decode-loop resident pointer table, then pass it to "
            "the persistent weight materializer before launching Qwen tasks."
        )
    if flags["persistent_weight_args_ready"]:
        return (
            "Materialize persistent task descriptors with resident weight "
            "pointers during the decode-loop runner."
        )
    if flags["weight_binding_ready"]:
        return (
            "Move the planned CUDA weight bindings into real persistent task "
            "arguments and full device residency."
        )
    if flags["metadata_validated"]:
        return (
            "Bind validated Qwen safetensors tensors to CUDA buffers and "
            "persistent-device task args."
        )
    if flags["shards_ready"]:
        return (
            "Run the safetensors metadata probe against the placed Qwen "
            "shards, then bind validated tensors to CUDA buffers."
        )
    return (
        "Keep the safetensors index and expected shape/dtype contract "
        "in sync with the shard placement and metadata probes."
    )


def shard_next_action(shards_ready: bool) -> str:
    if shards_ready:
        return (
            "Keep the local Qwen safetensors shards under tmp/sources and rerun "
            "the metadata probe when shard files change."
        )
    return (
        "Place or download the real Qwen safetensors shards under "
        "tmp/sources/qwen3-8b-safetensors, then rerun the metadata probe."
    )
