"""Qwen persistent task descriptor construction."""

from __future__ import annotations

from typing import Any

from .shape_contract import QWEN3_8B_TASK_SHAPE, QwenTaskShape, task_shape_fields


def layer_tensor(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.{suffix}.weight"


def tensor_arg_records(
    tensors: list[str],
    bindings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for index, tensor in enumerate(tensors):
        item = bindings.get(tensor)
        if item is None:
            records.append(
                {
                    "arg": f"tensor_args[{index}]",
                    "tensor": tensor,
                    "status": "missing_weight_binding",
                }
            )
            continue
        records.append(
            {
                "arg": f"tensor_args[{index}]",
                "slot_id": item["slot_id"],
                "tensor": tensor,
            }
        )
    return records


def runtime_tensor_arg_records(
    *,
    start_index: int,
    roles: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "arg": f"tensor_args[{index}]",
            "tensor": role,
            "role": role,
            "status": "runtime_generated_tensor",
            "device_ptr_source": f"runtime_buffers.{role}",
        }
        for index, role in enumerate(roles, start=start_index)
    ]


def tensor_arg_metadata(
    *,
    tensors: list[str],
    bindings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = []
    for index, tensor in enumerate(tensors):
        item = bindings.get(tensor)
        if item is None:
            continue
        record = {
            "arg": f"tensor_args[{index}]",
            "slot_id": item["slot_id"],
            "tensor": tensor,
        }
        for key in ("dtype", "shape", "size_bytes"):
            if key in item:
                record[key] = item[key]
        metadata.append(record)
    return metadata


def descriptor(
    *,
    descriptor_id: str,
    callable_name: str,
    phase: str,
    tensors: list[str],
    bindings: dict[str, dict[str, Any]],
    runtime_tensor_roles: list[str] | None = None,
    model_shape: QwenTaskShape = QWEN3_8B_TASK_SHAPE,
) -> dict[str, Any]:
    tensor_args = tensor_arg_records(tensors, bindings)
    if runtime_tensor_roles:
        tensor_args.extend(
            runtime_tensor_arg_records(
                start_index=len(tensor_args),
                roles=runtime_tensor_roles,
            )
        )
    missing = [
        item["tensor"]
        for item in tensor_args
        if item.get("status") == "missing_weight_binding"
    ]
    record = {
        "id": descriptor_id,
        "callable": callable_name,
        "phase": phase,
        "tensor_arg_count": len(tensor_args),
        "tensor_args": tensor_args,
        "status": "ready" if not missing else "missing_weight_binding",
        "missing_tensors": missing,
    }
    fields = task_shape_fields(callable_name, model_shape)
    if fields:
        record["task_shape_fields"] = fields
    metadata = tensor_arg_metadata(tensors=tensors, bindings=bindings)
    if metadata:
        record["tensor_arg_metadata"] = metadata
    return record


def layer_descriptors(
    *,
    layer: int,
    bindings: dict[str, dict[str, Any]],
    model_shape: QwenTaskShape = QWEN3_8B_TASK_SHAPE,
) -> list[dict[str, Any]]:
    prefix = f"layer_{layer}"
    return [
        descriptor(
            descriptor_id=f"{prefix}_input_norm",
            callable_name="qwen_rmsnorm_input",
            phase="per_layer_decode",
            tensors=[layer_tensor(layer, "input_layernorm")],
            bindings=bindings,
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_attention_qkv",
            callable_name="qwen_attention_qkv",
            phase="per_layer_decode",
            tensors=[
                layer_tensor(layer, "self_attn.q_proj"),
                layer_tensor(layer, "self_attn.k_proj"),
                layer_tensor(layer, "self_attn.v_proj"),
            ],
            bindings=bindings,
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_attention_qk_norm",
            callable_name="qwen_attention_qk_norm",
            phase="per_layer_decode",
            tensors=[
                layer_tensor(layer, "self_attn.q_norm"),
                layer_tensor(layer, "self_attn.k_norm"),
            ],
            bindings=bindings,
            runtime_tensor_roles=["rope_cos_table", "rope_sin_table"],
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_attention_o",
            callable_name="qwen_attention_o",
            phase="per_layer_decode",
            tensors=[layer_tensor(layer, "self_attn.o_proj")],
            bindings=bindings,
            runtime_tensor_roles=["kv_page_table"],
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_post_attention_norm",
            callable_name="qwen_rmsnorm_post_attention",
            phase="per_layer_decode",
            tensors=[layer_tensor(layer, "post_attention_layernorm")],
            bindings=bindings,
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_mlp_gate_up",
            callable_name="qwen_mlp_gate_up",
            phase="per_layer_decode",
            tensors=[
                layer_tensor(layer, "mlp.gate_proj"),
                layer_tensor(layer, "mlp.up_proj"),
            ],
            bindings=bindings,
            model_shape=model_shape,
        ),
        descriptor(
            descriptor_id=f"{prefix}_mlp_down",
            callable_name="qwen_mlp_down",
            phase="per_layer_decode",
            tensors=[layer_tensor(layer, "mlp.down_proj")],
            bindings=bindings,
            model_shape=model_shape,
        ),
    ]


def build_task_descriptors(
    *,
    bindings: dict[str, dict[str, Any]],
    num_hidden_layers: int,
    model_shape: QwenTaskShape = QWEN3_8B_TASK_SHAPE,
) -> list[dict[str, Any]]:
    descriptors = [
        descriptor(
            descriptor_id="embedding_lookup",
            callable_name="qwen_embedding_lookup",
            phase="prefill_or_decode_input",
            tensors=["model.embed_tokens.weight"],
            bindings=bindings,
            model_shape=model_shape,
        )
    ]
    for layer in range(num_hidden_layers):
        descriptors.extend(
            layer_descriptors(
                layer=layer,
                bindings=bindings,
                model_shape=model_shape,
            )
        )
    descriptors.extend(
        [
            descriptor(
                descriptor_id="final_norm",
                callable_name="qwen_final_norm",
                phase="per_token_decode",
                tensors=["model.norm.weight"],
                bindings=bindings,
                model_shape=model_shape,
            ),
            descriptor(
                descriptor_id="logits",
                callable_name="qwen_logits",
                phase="per_token_decode",
                tensors=["lm_head.weight"],
                bindings=bindings,
                model_shape=model_shape,
            ),
        ]
    )
    return descriptors
