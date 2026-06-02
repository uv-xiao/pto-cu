#!/usr/bin/env python3
"""Emit a PTO CUDA persistent-device Qwen serving lifecycle plan."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}


@dataclass(frozen=True)
class QwenModelShape:
    model_id: str
    config_revision: str
    dtype: str
    dtype_bytes: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float


QWEN3_8B_SHAPE = QwenModelShape(
    model_id="Qwen/Qwen3-8B",
    config_revision="d117af2f304f02a8647f88fe05b61cfb405a1d9e",
    dtype="bfloat16",
    dtype_bytes=2,
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    max_position_embeddings=40960,
    rope_theta=1000000.0,
)


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


def model_shape_payload(shape: QwenModelShape) -> dict[str, Any]:
    return {
        "model_id": shape.model_id,
        "config_revision": shape.config_revision,
        "dtype": shape.dtype,
        "dtype_bytes": shape.dtype_bytes,
        "hidden_size": shape.hidden_size,
        "num_hidden_layers": shape.num_hidden_layers,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "vocab_size": shape.vocab_size,
        "max_position_embeddings": shape.max_position_embeddings,
        "rope_theta": shape.rope_theta,
        "source_note": (
            "tmp/sources/qwen3-8b-config-d117af2f.json mirrors the "
            "Hugging Face Qwen/Qwen3-8B config snapshot used for this plan."
        ),
    }


def kv_cache_plan(
    *,
    shape: QwenModelShape,
    workload_id: str,
    prompt_tokens: int,
    decode_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    sequence_capacity = prompt_tokens + decode_tokens
    elements = (
        batch_size
        * sequence_capacity
        * shape.num_hidden_layers
        * 2
        * shape.num_key_value_heads
        * shape.head_dim
    )
    return {
        "workload_id": workload_id,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "sequence_capacity_tokens": sequence_capacity,
        "layout": "contiguous[layer][kv][batch][token][kv_head][head_dim]",
        "element_dtype": shape.dtype,
        "element_count": elements,
        "bytes": elements * shape.dtype_bytes,
        "token_position_lifecycle": [
            "prefill writes positions [0, prompt_tokens)",
            "decode step t reads [0, prompt_tokens + t)",
            "decode step t writes position prompt_tokens + t",
            "runner stops after decode_tokens or EOS policy",
        ],
    }


def persistent_task_mapping() -> list[dict[str, Any]]:
    return [
        {
            "callable": "qwen_embed_token",
            "phase": "prefill_and_decode",
            "inputs": ["input_ids", "token_position"],
            "outputs": ["hidden_state"],
            "status": "planned_kernel_body",
        },
        {
            "callable": "qwen_layer_attention",
            "phase": "per_layer_decode",
            "inputs": [
                "hidden_state",
                "qkv_weights",
                "rope_tables",
                "kv_cache_read_window",
            ],
            "outputs": ["kv_cache_write_slot", "attention_output"],
            "status": "planned_kernel_body",
        },
        {
            "callable": "qwen_layer_mlp",
            "phase": "per_layer_decode",
            "inputs": ["hidden_state", "gate_up_down_weights"],
            "outputs": ["hidden_state"],
            "status": "planned_kernel_body",
        },
        {
            "callable": "qwen_logits_and_argmax",
            "phase": "per_token_decode",
            "inputs": ["hidden_state", "lm_head_or_tied_embeddings"],
            "outputs": ["next_token_id"],
            "status": "planned_kernel_body",
        },
    ]


def weight_binding_plan(shape: QwenModelShape) -> list[dict[str, Any]]:
    return [
        {
            "name": "embedding",
            "tensor_count": 1,
            "shape": [shape.vocab_size, shape.hidden_size],
            "binding": "read_only_device_tensor",
            "status": "planned_loader_binding",
        },
        {
            "name": "attention_qkv_o",
            "tensor_count": shape.num_hidden_layers * 4,
            "shape": "per-layer q_proj, k_proj, v_proj, o_proj",
            "binding": "read_only_device_tensor",
            "status": "planned_loader_binding",
        },
        {
            "name": "mlp_gate_up_down",
            "tensor_count": shape.num_hidden_layers * 3,
            "shape": "per-layer gate_proj, up_proj, down_proj",
            "binding": "read_only_device_tensor",
            "status": "planned_loader_binding",
        },
        {
            "name": "norm_and_logits",
            "tensor_count": 1 + shape.num_hidden_layers * 2,
            "shape": "final norm plus per-layer input/post-attention norms",
            "binding": "read_only_device_tensor",
            "status": "planned_loader_binding",
        },
    ]


def workload_plans(shape: QwenModelShape) -> list[dict[str, Any]]:
    payload = load_json(VIEWER_DATA / "serving_workloads.json")
    plans: list[dict[str, Any]] = []
    for workload in payload.get("serving_workloads", []):
        if workload.get("id") not in TARGET_WORKLOAD_IDS:
            continue
        prompt_policy = workload.get("prompt_policy", {})
        decode_policy = workload.get("decode_policy", {})
        prompt_tokens = int(prompt_policy.get("target_prompt_tokens", 0))
        decode_tokens = int(decode_policy.get("decode_tokens", 0))
        batch_sizes = [int(item) for item in decode_policy.get("batch_sizes", [])]
        plans.append(
            {
                "id": workload.get("id"),
                "primary_model": workload.get("model_policy", {}).get(
                    "primary_model"
                ),
                "prompt_tokens": prompt_tokens,
                "decode_tokens": decode_tokens,
                "batch_sizes": batch_sizes,
                "kv_cache_plans": [
                    kv_cache_plan(
                        shape=shape,
                        workload_id=str(workload.get("id")),
                        prompt_tokens=prompt_tokens,
                        decode_tokens=decode_tokens,
                        batch_size=batch_size,
                    )
                    for batch_size in batch_sizes
                ],
            }
        )
    return plans


def build_lifecycle_plan() -> dict[str, Any]:
    shape = QWEN3_8B_SHAPE
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_serving_lifecycle_plan",
        "status": "partial_runtime_plan",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "runtime": "cuda/persistent_device",
        "model_shape": model_shape_payload(shape),
        "workload_plans": workload_plans(shape),
        "weight_binding_plan": weight_binding_plan(shape),
        "persistent_task_mapping": persistent_task_mapping(),
        "implemented_contracts": [
            "qwen3_8b_model_shape",
            "serving_policy_to_kv_cache_plan",
            "persistent_device_task_mapping",
        ],
        "remaining_runtime_gaps": [
            "runtime_token_id_binding",
            "safetensors_weight_loader",
            "cuda_device_allocation_and_binding",
            "generated_qwen_kernel_bodies",
            "decode_loop_execution",
            "viewer_result_import",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_lifecycle_plan()
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
