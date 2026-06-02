"""Qwen model-shape tensor-tile task body contracts."""

from __future__ import annotations

import hashlib
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentTaskFunction,
    render_persistent_dag_source,
)


FUNC_ID_QWEN_TENSOR_BASE = 7240

_WMMA_TEMPLATE = """
if (task->rows != {rows}U || task->cols != {cols}U || task->inner != {inner}U) {{
  return;
}}
using namespace nvcuda;
unsigned long long tile_count = task->n / task->out_batch_stride;
for (unsigned long long tile_id = 0; tile_id < tile_count; ++tile_id) {{
  unsigned long long a_base = tile_id * task->a_batch_stride;
  unsigned long long b_base = tile_id * task->b_batch_stride;
  unsigned long long out_base = tile_id * task->out_batch_stride;
  for (unsigned int col = 0; col < {cols}U; col += 16U) {{
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    for (unsigned int k = 0; k < {inner}U; k += 8U) {{
      wmma::load_matrix_sync(a_frag, task->a + a_base + k, task->lda);
      wmma::load_matrix_sync(b_frag, task->b + b_base + k * task->ldb + col, task->ldb);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }}
    wmma::store_matrix_sync(task->out + out_base + col, acc_frag, task->ldc, wmma::mem_row_major);
  }}
}}
""".strip()


def qwen_tensor_tile_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen_attention_projection_tile",
            "callable": "qwen_attention_projection_wmma_tile",
            "func_id": FUNC_ID_QWEN_TENSOR_BASE,
            "phase": "per_layer_decode",
            "tensor_tile": {"rows": 16, "cols": 64, "inner": 128},
            "model_mapping": "Qwen attention projection tile.",
            "status": "source_contract_ready",
        },
        {
            "id": "qwen_mlp_projection_tile",
            "callable": "qwen_mlp_projection_wmma_tile",
            "func_id": FUNC_ID_QWEN_TENSOR_BASE + 1,
            "phase": "per_layer_decode",
            "tensor_tile": {"rows": 16, "cols": 64, "inner": 256},
            "model_mapping": "Qwen MLP projection tile.",
            "status": "source_contract_ready",
        },
    ]


def qwen_tensor_tile_task_functions() -> list[CudaPersistentTaskFunction]:
    functions: list[CudaPersistentTaskFunction] = []
    for spec in qwen_tensor_tile_specs():
        tile = spec["tensor_tile"]
        functions.append(
            CudaPersistentTaskFunction(
                func_id=spec["func_id"],
                name=spec["callable"],
                threading="block",
                body=_WMMA_TEMPLATE.format(**tile),
            )
        )
    return functions


def build_qwen_tensor_tile_contract() -> dict[str, Any]:
    source = render_persistent_dag_source(qwen_tensor_tile_task_functions())
    return {
        "status": "qwen_tensor_tile_source_contract_ready",
        "runtime": "cuda/persistent_device",
        "task_functions": qwen_tensor_tile_specs(),
        "wmma": {
            "api": "nvcuda::wmma",
            "mma_shape": "m16n16k8",
            "input": "tf32",
            "accumulator": "f32",
        },
        "rendered_source": {
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "line_count": len(source.splitlines()),
            "required_fragments": [
                "task->rows != 16U",
                "task->cols != 64U",
                "k < 128U",
                "k < 256U",
                "wmma::mma_sync",
            ],
        },
        "remaining_wiring": [
            "route model-shape benchmark descriptors to these Qwen func_ids",
            "capture multi-repeat A100/H200 throughput rows",
        ],
    }
