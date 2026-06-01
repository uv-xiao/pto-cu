"""Load scaffold input artifacts from neighboring CUDA examples."""

from __future__ import annotations

from typing import Any

from .common import (
    CUDA_TOKEN_BUFFER_BINDING,
    CUDA_WEIGHT_BINDING,
    DECODE_LOOP_RUNNER,
    KV_CACHE_BINDING,
    LIFECYCLE_PLAN,
    PERSISTENT_DECODE_ARGS,
    PERSISTENT_WEIGHT_ARGS,
    PERSISTENT_WEIGHT_MATERIALIZATION,
    PROMPT_ACCOUNTING,
    RESIDENT_WEIGHT_TABLE,
    RUNTIME_INPUT_BINDING,
    SAFETENSORS_FETCH,
    SAFETENSORS_METADATA,
    TASK_BODIES,
    TOKEN_POINTER_TABLE,
    WEIGHT_INVENTORY,
    load_python_module,
    load_python_payload,
)


def load_lifecycle_plan() -> dict[str, Any]:
    return load_python_payload(
        LIFECYCLE_PLAN, "qwen_serving_lifecycle_plan", "build_lifecycle_plan"
    )


def load_prompt_accounting() -> dict[str, Any]:
    return load_python_payload(
        PROMPT_ACCOUNTING, "qwen_prompt_accounting", "build_prompt_accounting"
    )


def load_runtime_input_binding() -> dict[str, Any]:
    return load_python_payload(
        RUNTIME_INPUT_BINDING,
        "qwen_runtime_input_binding",
        "build_runtime_input_binding",
    )


def load_cuda_token_buffer_binding() -> dict[str, Any]:
    module = load_python_module(
        CUDA_TOKEN_BUFFER_BINDING, "qwen_cuda_token_buffer_binding"
    )
    return {} if module is None else module.build_cuda_token_buffer_binding(no_cuda_probe=True)


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
        WEIGHT_INVENTORY, "qwen_weight_inventory", "build_weight_inventory"
    )


def load_safetensors_shards() -> dict[str, Any]:
    return load_python_payload(
        SAFETENSORS_FETCH, "qwen_safetensors_fetch", "build_shard_status"
    )


def load_safetensors_metadata() -> dict[str, Any]:
    return load_python_payload(
        SAFETENSORS_METADATA,
        "qwen_safetensors_metadata",
        "build_metadata_probe",
    )


def load_cuda_weight_binding() -> dict[str, Any]:
    module = load_python_module(CUDA_WEIGHT_BINDING, "qwen_cuda_weight_binding")
    return {} if module is None else module.build_weight_binding(no_cuda_probe=True)


def load_persistent_weight_args() -> dict[str, Any]:
    module = load_python_module(PERSISTENT_WEIGHT_ARGS, "qwen_persistent_weight_args")
    return {} if module is None else module.build_weight_arg_manifest()


def load_persistent_weight_materialization() -> dict[str, Any]:
    module = load_python_module(
        PERSISTENT_WEIGHT_MATERIALIZATION,
        "qwen_persistent_weight_materialization",
    )
    return {} if module is None else module.build_materialization_manifest()


def load_resident_weight_table() -> dict[str, Any]:
    module = load_python_module(RESIDENT_WEIGHT_TABLE, "qwen_resident_weight_table")
    return {} if module is None else module.build_resident_table_lifecycle()


def load_kv_cache_binding() -> dict[str, Any]:
    return load_python_payload(
        KV_CACHE_BINDING, "qwen_kv_cache_binding", "build_kv_cache_lifecycle"
    )


def load_decode_loop_runner() -> dict[str, Any]:
    return load_python_payload(
        DECODE_LOOP_RUNNER, "qwen_decode_loop_runner", "build_decode_loop_runner"
    )


def load_task_bodies() -> dict[str, Any]:
    return load_python_payload(
        TASK_BODIES,
        "qwen_persistent_task_bodies",
        "build_task_body_manifest",
    )


def load_artifacts() -> dict[str, dict[str, Any]]:
    return {
        "lifecycle_plan": load_lifecycle_plan(),
        "prompt_accounting": load_prompt_accounting(),
        "runtime_input_binding": load_runtime_input_binding(),
        "cuda_token_buffer_binding": load_cuda_token_buffer_binding(),
        "persistent_decode_args": load_persistent_decode_args(),
        "token_pointer_table": load_token_pointer_table(),
        "weight_inventory": load_weight_inventory(),
        "safetensors_shards": load_safetensors_shards(),
        "safetensors_metadata": load_safetensors_metadata(),
        "cuda_weight_binding": load_cuda_weight_binding(),
        "persistent_weight_args": load_persistent_weight_args(),
        "persistent_weight_materialization": load_persistent_weight_materialization(),
        "resident_weight_table": load_resident_weight_table(),
        "kv_cache_binding": load_kv_cache_binding(),
        "decode_loop_runner": load_decode_loop_runner(),
        "task_bodies": load_task_bodies(),
    }
