"""Build a reviewable manifest for generated Qwen persistent task bodies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentTaskFunction,
    render_persistent_dag_source,
)


ROOT = Path(__file__).resolve().parents[3]
ENTRY_NAME = "pto_persistent_dag_f32_executor"
SOURCE_KIND = "generated-dispatch"
FUNC_ID_BASE = 7100


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def body_specs() -> list[dict[str, Any]]:
    return [
        {
            "callable": "qwen_embedding_lookup",
            "phase": "prefill_or_decode_input",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["input_ids", "embedding_weight"],
            "body": """
const unsigned int token_id =
    reinterpret_cast<const unsigned int *>(task->a)[i % task->n];
const float *embedding = task->tensor_args[0];
task->out[i] = embedding ? embedding[token_id & 3U] : 0.0f;
""",
        },
        {
            "callable": "qwen_rmsnorm_input",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "input_layernorm_weight"],
            "body": """
const float *weight = task->tensor_args[0];
const float scale = weight ? weight[i & 3U] : 1.0f;
task->out[i] = task->a[i] * scale;
""",
        },
        {
            "callable": "qwen_attention_qkv",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "b", "out", "c", "d", "tensor_args"],
            "consumes_roles": [
                "hidden_state",
                "attention_mask",
                "key_cache",
                "value_cache",
                "key_cache_writeback",
                "value_cache_writeback",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ],
            "body": """
const float mask = task->b ? task->b[i % task->n] : 1.0f;
const unsigned long long kv_index = i % task->n;
const float key = task->c ? task->c[kv_index] : 0.0f;
const float value = task->d ? task->d[kv_index] : 0.0f;
const float *q_proj = task->tensor_args[0];
const float q = q_proj ? q_proj[i & 3U] : 0.0f;
task->out[i] = task->a[i] + mask + key + value + q;
if (task->c) {
    task->c[kv_index] = task->a[i] + q;
}
if (task->d) {
    task->d[kv_index] = task->out[i];
}
""",
        },
        {
            "callable": "qwen_attention_qk_norm",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["q_state", "q_norm_weight", "k_norm_weight"],
            "body": """
const float *q_norm = task->tensor_args[0];
const float *k_norm = task->tensor_args[1];
const float q = q_norm ? q_norm[i & 3U] : 1.0f;
const float k = k_norm ? k_norm[i & 3U] : 1.0f;
task->out[i] = task->a[i] * 0.5f * (q + k);
""",
        },
        {
            "callable": "qwen_attention_o",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["attention_state", "o_proj_weight"],
            "body": """
const float *weight = task->tensor_args[0];
task->out[i] = task->a[i] + (weight ? weight[i & 3U] : 0.0f);
""",
        },
        {
            "callable": "qwen_rmsnorm_post_attention",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "post_attention_layernorm_weight"],
            "body": """
const float *weight = task->tensor_args[0];
task->out[i] = task->a[i] * (weight ? weight[i & 3U] : 1.0f);
""",
        },
        {
            "callable": "qwen_mlp_gate_up",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "gate_proj_weight", "up_proj_weight"],
            "body": """
const float *gate = task->tensor_args[0];
const float *up = task->tensor_args[1];
const float gated = task->a[i] * (gate ? gate[i & 3U] : 1.0f);
task->out[i] = gated + (up ? up[i & 3U] : 0.0f);
""",
        },
        {
            "callable": "qwen_mlp_down",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["mlp_state", "down_proj_weight"],
            "body": """
const float *down = task->tensor_args[0];
task->out[i] = task->a[i] + (down ? down[i & 3U] : 0.0f);
""",
        },
        {
            "callable": "qwen_final_norm",
            "phase": "per_token_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "final_norm_weight"],
            "body": """
const float *weight = task->tensor_args[0];
task->out[i] = task->a[i] * (weight ? weight[i & 3U] : 1.0f);
""",
        },
        {
            "callable": "qwen_logits",
            "phase": "per_token_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "lm_head_weight", "output_ids"],
            "body": """
const float *lm_head = task->tensor_args[0];
const float logit = task->a[i] + (lm_head ? lm_head[i & 3U] : 0.0f);
task->out[i] = logit;
""",
        },
    ]


def task_functions() -> list[CudaPersistentTaskFunction]:
    return [
        CudaPersistentTaskFunction(
            func_id=FUNC_ID_BASE + index,
            name=spec["callable"],
            body=spec["body"],
        )
        for index, spec in enumerate(body_specs())
    ]


def source_preview(source: str, callables: list[str]) -> str:
    lines = source.splitlines()
    selected: list[str] = []
    for callable_name in callables:
        marker = f"__device__ void pto_task_{callable_name}"
        for line_index, line in enumerate(lines):
            if marker not in line:
                continue
            selected.extend(lines[line_index : line_index + 14])
            selected.append("")
            break
    return "\n".join(selected).strip()


def build_task_body_manifest(num_hidden_layers: int = 36) -> dict[str, Any]:
    specs = body_specs()
    functions = task_functions()
    source = render_persistent_dag_source(functions)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    callables = [spec["callable"] for spec in specs]
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_task_bodies",
        "status": "generated_task_bodies_ready",
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "source_kind": SOURCE_KIND,
        "num_hidden_layers": num_hidden_layers,
        "task_body_count": len(specs),
        "task_bodies": [
            {
                "func_id": function.func_id,
                "callable": spec["callable"],
                "phase": spec["phase"],
                "consumes_fields": spec["consumes_fields"],
                "consumes_roles": spec["consumes_roles"],
            }
            for function, spec in zip(functions, specs, strict=True)
        ],
        "coverage": {
            "token_fields": ["a", "b", "out"],
            "kv_fields": ["c", "d"],
            "kv_write_policy": "mutable_kv_fields_ready",
            "weight_fields": ["tensor_args"],
            "descriptor_source": "examples/cuda/qwen_persistent_weight_args.py",
            "decode_argument_source": "examples/cuda/qwen_persistent_decode_args.py",
        },
        "rendered_source": {
            "entry_name": ENTRY_NAME,
            "sha256": source_sha256,
            "line_count": len(source.splitlines()),
            "preview": source_preview(
                source,
                [
                    "qwen_attention_qkv",
                    "qwen_mlp_gate_up",
                    "qwen_logits",
                ],
            ),
        },
        "implemented_contracts": [
            "generated_qwen_kernel_bodies",
            "qwen_kernel_token_field_consumption",
            "qwen_kernel_kv_field_consumption",
            "qwen_kernel_kv_cache_writeback_field_contract",
            "qwen_kernel_weight_tensor_arg_consumption",
        ],
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "cuda_live_decode_loop_execution",
            "viewer_result_import",
        ],
    }


def write_source(path: Path) -> dict[str, Any]:
    source = render_persistent_dag_source(task_functions())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return {
        "path": repo_relative(path),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "line_count": len(source.splitlines()),
    }
