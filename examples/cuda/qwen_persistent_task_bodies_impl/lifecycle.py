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

from .logits_feedback import qwen_logits_spec
from .kernel_source_map import build_kernel_source_map
from .oracle import build_numeric_oracle, build_qwen_unit_math_oracle
from .tensor_tiles import build_qwen_tensor_tile_contract


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
task->out[i] = pto_cuda_tensor_arg_f32(task, 0U, token_id & 3U, 0.0f);
""",
        },
        {
            "callable": "qwen_rmsnorm_input",
            "phase": "per_layer_decode",
            "threading": "block",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "input_layernorm_weight"],
            "body": """
if (task->scalar_arg_count == 0) {
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const float scale = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
        task->out[i] = task->a[i] * scale;
    }
} else if (task->scalar_arg_count > 1) {
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const float norm_weight =
            pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
        task->out[i] = task->a[i] * task->scalar_args[1] * norm_weight;
    }
} else {
    __shared__ float partial[1024];
    float mean_square = 0.0f;
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        mean_square += task->a[j] * task->a[j];
    }
    partial[threadIdx.x] = mean_square;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0U; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale =
        rsqrtf(partial[0] / static_cast<float>(task->n) + 0.000001f);
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const float norm_weight =
            pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
        task->out[i] = task->a[i] * scale * norm_weight;
    }
}
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
const bool has_projection_weights =
    task->tensor_arg_count >= 3U && task->tensor_args[0] &&
    task->tensor_args[1] && task->tensor_args[2];
if (task->scalar_arg_count > 0 && has_projection_weights) {
    const float q =
        task->a[i] * pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    const float k =
        task->a[i] * pto_cuda_tensor_arg_f32(task, 1U, i & 3U, 0.0f);
    const float v =
        task->a[i] * pto_cuda_tensor_arg_f32(task, 2U, i & 3U, 0.0f);
    if (task->c) {
        task->c[kv_index] = k;
    }
    if (task->d) {
        task->d[kv_index] = v;
    }
    task->out[i] = v;
} else {
    const float q = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    task->out[i] = task->a[i] + mask + key + value + q;
    if (task->c) {
        task->c[kv_index] = task->a[i] + q;
    }
    if (task->d) {
        task->d[kv_index] = task->out[i];
    }
}
""",
        },
        {
            "callable": "qwen_attention_qk_norm",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["q_state", "q_norm_weight", "k_norm_weight"],
            "body": """
const float q = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
const float k = pto_cuda_tensor_arg_f32(task, 1U, i & 3U, 1.0f);
task->out[i] = task->a[i] * 0.5f * (q + k);
""",
        },
        {
            "callable": "qwen_attention_o",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["attention_state", "o_proj_weight"],
            "body": """
const float weight = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
task->out[i] = task->a[i] + weight;
""",
        },
        {
            "callable": "qwen_rmsnorm_post_attention",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "post_attention_layernorm_weight"],
            "body": """
const float weight = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
task->out[i] = task->a[i] * weight;
""",
        },
        {
            "callable": "qwen_mlp_gate_up",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "gate_proj_weight", "up_proj_weight"],
            "body": """
const float gate_value = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, task->a[i]);
const float up_value = pto_cuda_tensor_arg_f32(task, 1U, i & 3U, task->a[i]);
const float silu_gate = gate_value / (1.0f + expf(-gate_value));
task->out[i] = silu_gate * up_value;
""",
        },
        {
            "callable": "qwen_mlp_down",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["mlp_state", "down_proj_weight"],
            "body": """
const float down = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
task->out[i] = task->a[i] + down;
""",
        },
        {
            "callable": "qwen_final_norm",
            "phase": "per_token_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "final_norm_weight"],
            "body": """
const float weight = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
task->out[i] = task->a[i] * weight;
""",
        },
        qwen_logits_spec(),
    ]


def task_functions() -> list[CudaPersistentTaskFunction]:
    return [
        CudaPersistentTaskFunction(
            func_id=FUNC_ID_BASE + index,
            name=spec["callable"],
            body=spec["body"],
            threading=spec.get("threading", "element"),
        )
        for index, spec in enumerate(body_specs())
    ]


def source_preview(source: str, callables: list[str], *, window_lines: int = 44) -> str:
    lines = source.splitlines()
    selected: list[str] = []
    for callable_name in callables:
        marker = f"__device__ void pto_task_{callable_name}"
        for line_index, line in enumerate(lines):
            if marker not in line:
                continue
            selected.extend(lines[line_index : line_index + window_lines])
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
                "threading": function.threading,
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
            "decode_feedback_fields": ["tensor_args[2]", "tensor_args[3]"],
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
                    "qwen_rmsnorm_input",
                    "qwen_attention_qkv",
                    "qwen_mlp_gate_up",
                    "qwen_logits",
                ],
            ),
        },
        "numeric_oracle": build_numeric_oracle(callables),
        "qwen_unit_math_oracle": build_qwen_unit_math_oracle(),
        "qwen_tensor_tile_contract": build_qwen_tensor_tile_contract(),
        "qwen_kernel_source_map": build_kernel_source_map(),
        "implemented_contracts": [
            "generated_qwen_kernel_bodies",
            "controlled_proxy_numeric_oracle",
            "qwen_unit_math_oracle",
            "qwen_tensor_tile_source_contract",
            "qwen_kernel_source_map",
            "qwen_unit_math_source_coverage",
            "qwen_kernel_token_field_consumption",
            "qwen_kernel_kv_field_consumption",
            "qwen_kernel_kv_cache_writeback_field_contract",
            "qwen_kernel_weight_tensor_arg_consumption",
            "qwen_logits_device_sampled_token_feedback_source",
        ],
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "cuda_live_qwen_unit_math_execution",
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
