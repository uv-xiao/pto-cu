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
from .oracle import (
    build_numeric_oracle,
    build_qwen_decode_attention_oracle,
    build_qwen_unit_math_oracle,
)
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
if (task->cols > 0U && task->tensor_arg_count > 0U && task->tensor_args[0]) {
    const unsigned int token_row =
        static_cast<unsigned int>(i / task->cols);
    const unsigned int hidden_col =
        static_cast<unsigned int>(i % task->cols);
    const unsigned int prompt_stride =
        task->a_batch_stride > 0U ? task->a_batch_stride : 1U;
    const unsigned int requested_token_position =
        task->scalar_arg_count > 2U ?
        static_cast<unsigned int>(task->scalar_args[2]) : 0U;
    const unsigned int token_position = prompt_stride > 0U ?
        requested_token_position % prompt_stride : 0U;
    const unsigned int embedding_stride =
        task->ldb > 0U ? task->ldb : task->cols;
    const unsigned int token_id =
        reinterpret_cast<const unsigned int *>(task->a)[
            static_cast<unsigned long long>(token_row) * prompt_stride +
            token_position];
    const unsigned long long embedding_weight_index =
        static_cast<unsigned long long>(token_id) * embedding_stride +
        hidden_col;
    task->out[i] =
        pto_cuda_tensor_arg_f32(task, 0U, embedding_weight_index, 0.0f);
} else {
const unsigned int token_id =
    reinterpret_cast<const unsigned int *>(task->a)[i % task->n];
task->out[i] = pto_cuda_tensor_arg_f32(task, 0U, token_id & 3U, 0.0f);
}
""",
        },
        {
            "callable": "qwen_rmsnorm_input",
            "phase": "per_layer_decode",
            "threading": "block",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "input_layernorm_weight"],
            "body": """
if (task->scalar_arg_count > 1 && task->scalar_args[0] == 1.0f &&
    task->scalar_args[1] == 0.0f && task->cols > 0U) {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int row = static_cast<unsigned int>(j / task->cols);
        const unsigned int col = static_cast<unsigned int>(j % task->cols);
        const unsigned int input_stride =
            task->lda > 0U ? task->lda : task->cols;
        const unsigned long long row_base =
            static_cast<unsigned long long>(row) * input_stride;
        float mean_square = 0.0f;
        for (unsigned int k = 0U; k < task->cols; ++k) {
            const float value = task->a[row_base + k];
            mean_square += value * value;
        }
        const float scale =
            rsqrtf(mean_square / static_cast<float>(task->cols) + 0.000001f);
        const float norm_weight =
            pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
        task->out[j] = task->a[row_base + col] * scale * norm_weight;
    }
} else if (task->scalar_arg_count == 0) {
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const float scale = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 1.0f);
        task->out[i] = task->a[i] * scale;
    }
} else if (task->scalar_arg_count > 1) {
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const unsigned int col = static_cast<unsigned int>(
            task->cols > 0U ? i % task->cols : i);
        const float norm_weight =
            pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
        task->out[i] = task->a[i] * task->scalar_args[1] * norm_weight;
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
const bool has_projection_weights =
    task->tensor_arg_count >= 3U && task->tensor_args[0] &&
    task->tensor_args[1] && task->tensor_args[2];
if (task->cols > 0U && task->inner > 0U && has_projection_weights) {
    const unsigned int row = static_cast<unsigned int>(i / task->cols);
    const unsigned int col = static_cast<unsigned int>(i % task->cols);
    const unsigned int requested_active_projection_cols =
        task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1) :
        task->cols;
    const unsigned int active_projection_cols =
        requested_active_projection_cols < task->cols ?
        requested_active_projection_cols : task->cols;
    const unsigned int q_width = task->inner;
    const unsigned int kv_width =
        task->cols > q_width ? (task->cols - q_width) / 2U : q_width;
    const unsigned int kv_page_size =
        task->scalar0 > 0.0f ? static_cast<unsigned int>(task->scalar0) : 16U;
    const unsigned int *kv_page_table =
        task->tensor_arg_count > 3U && task->tensor_args[3] ?
        reinterpret_cast<const unsigned int *>(task->tensor_args[3]) : nullptr;
    const unsigned int decode_position = task->scalar_arg_count > 2U ?
        static_cast<unsigned int>(task->scalar_args[2]) : row;
    const unsigned int sequence_capacity =
        task->b_batch_stride > 0U ? task->b_batch_stride : kv_page_size;
    const unsigned int cache_batch_size =
        task->out_batch_stride > 0U ? task->out_batch_stride : task->rows;
    const unsigned int kv_layer_index = task->scalar_arg_count > 3U ?
        static_cast<unsigned int>(task->scalar_args[3]) : 0U;
    const unsigned long long kv_layer_base =
        static_cast<unsigned long long>(kv_layer_index) * cache_batch_size *
        sequence_capacity * kv_width;
    float projected = 0.0f;
    if (col >= active_projection_cols) {
        task->out[i] = 0.0f;
    } else if (col < q_width) {
        projected = pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f);
        task->out[i] = projected;
    } else if (col < q_width + kv_width) {
        const unsigned int kv_col = col - q_width;
        projected = pto_cuda_linear_arg_f32(task, 1U, row, kv_col, 0.0f);
        const unsigned int logical_page = decode_position / kv_page_size;
        const unsigned int page_offset = decode_position % kv_page_size;
        const unsigned int physical_page = kv_page_table ?
            kv_page_table[logical_page] : logical_page;
        const unsigned long long token_slot =
            static_cast<unsigned long long>(physical_page) * kv_page_size +
            page_offset;
        const unsigned long long kv_write_index =
            kv_layer_base +
            static_cast<unsigned long long>(row) * sequence_capacity * kv_width +
            token_slot * kv_width + kv_col;
        if (task->c) {
            task->c[kv_write_index] = projected;
        }
        task->out[i] = projected;
    } else {
        const unsigned int kv_col = col - q_width - kv_width;
        projected = pto_cuda_linear_arg_f32(task, 2U, row, kv_col, 0.0f);
        const unsigned int logical_page = decode_position / kv_page_size;
        const unsigned int page_offset = decode_position % kv_page_size;
        const unsigned int physical_page = kv_page_table ?
            kv_page_table[logical_page] : logical_page;
        const unsigned long long token_slot =
            static_cast<unsigned long long>(physical_page) * kv_page_size +
            page_offset;
        const unsigned long long kv_write_index =
            kv_layer_base +
            static_cast<unsigned long long>(row) * sequence_capacity * kv_width +
            token_slot * kv_width + kv_col;
        if (task->d) {
            task->d[kv_write_index] = projected;
        }
        task->out[i] = projected;
    }
} else if (task->scalar_arg_count > 0 && has_projection_weights) {
    const float q = task->a[i] *
        pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    const float k = task->a[i] *
        pto_cuda_tensor_arg_f32(task, 1U, i & 3U, 0.0f);
    const float v = task->a[i] *
        pto_cuda_tensor_arg_f32(task, 2U, i & 3U, 0.0f);
    const unsigned long long kv_index = i % task->n;
    if (task->c) {
        task->c[kv_index] = k;
    }
    if (task->d) {
        task->d[kv_index] = v;
    }
    task->out[i] = v;
} else {
    const float mask = task->b ? task->b[i % task->n] : 1.0f;
    const unsigned long long kv_index = i % task->n;
    const float key = task->c ? task->c[kv_index] : 0.0f;
    const float value = task->d ? task->d[kv_index] : 0.0f;
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
            "threading": "block",
            "consumes_fields": ["a", "out", "c", "tensor_args", "scalar_args"],
            "consumes_roles": [
                "q_state",
                "q_norm_weight",
                "k_norm_weight",
                "rope_cos_table",
                "rope_sin_table",
                "kv_page_table",
            ],
            "body": """
if (task->cols > 0U && task->inner > 0U && task->tensor_arg_count >= 2U &&
    task->tensor_args[0] && task->tensor_args[1]) {
    const unsigned int qk_norm_input_stride =
        task->a_batch_stride > 0U ? task->a_batch_stride : task->cols;
    const unsigned int head_dim = task->inner;
    const unsigned int query_heads = task->rows > 0U ? task->rows : 1U;
    const unsigned int kv_heads = task->ldb > 0U ? task->ldb : query_heads;
    const unsigned int q_width = query_heads * head_dim;
    const unsigned int kv_width = kv_heads * head_dim;
    const unsigned int raw_kv_page_size =
        task->scalar0 > 0.0f ? static_cast<unsigned int>(task->scalar0) : 16U;
    const unsigned int kv_page_size =
        raw_kv_page_size > 0U ? raw_kv_page_size : 16U;
    const unsigned int decode_position = task->scalar_arg_count > 2U ?
        static_cast<unsigned int>(task->scalar_args[2]) : 0U;
    const unsigned int sequence_capacity =
        task->b_batch_stride > 0U ? task->b_batch_stride : kv_page_size;
    const unsigned int cache_batch_size =
        task->out_batch_stride > 0U ? task->out_batch_stride : task->rows;
    const unsigned int kv_layer_index = task->scalar_arg_count > 3U ?
        static_cast<unsigned int>(task->scalar_args[3]) : 0U;
    const unsigned long long kv_layer_base =
        static_cast<unsigned long long>(kv_layer_index) * cache_batch_size *
        sequence_capacity * kv_width;
    const unsigned int *qk_norm_kv_page_table =
        task->tensor_arg_count > 4U && task->tensor_args[4] ?
        reinterpret_cast<const unsigned int *>(task->tensor_args[4]) : nullptr;
    if (task->cols >= q_width + kv_width) {
        for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
            const unsigned int row = static_cast<unsigned int>(j / task->cols);
            const unsigned long long row_base =
                static_cast<unsigned long long>(row) * qk_norm_input_stride;
            const unsigned int col = static_cast<unsigned int>(j % task->cols);
            if (col >= q_width + kv_width) {
                task->out[j] = 0.0f;
                continue;
            }
            const bool is_query_region = col < q_width;
            const unsigned int region_col =
                is_query_region ? col : col - q_width;
            const unsigned int source_col = is_query_region ?
                col : q_width + region_col;
            const unsigned int norm_slot = is_query_region ? 0U : 1U;
            const unsigned int head_base =
                source_col - (source_col % head_dim);
            float mean_square = 0.0f;
            for (unsigned int k = 0U; k < head_dim; ++k) {
                const float value = task->a[row_base + head_base + k];
                mean_square += value * value;
            }
            const float scale =
                rsqrtf(mean_square / static_cast<float>(head_dim) + 0.000001f);
            const unsigned int head_col = source_col % head_dim;
            const float norm_weight =
                pto_cuda_tensor_arg_f32(task, norm_slot, head_col, 1.0f);
            const float normalized =
                task->a[row_base + source_col] * scale * norm_weight;
            const unsigned int half_head_dim = head_dim >> 1U;
            if (half_head_dim > 0U) {
                const bool first_half = head_col < half_head_dim;
                const unsigned int pair_head_col = first_half ?
                    head_col + half_head_dim : head_col - half_head_dim;
                const unsigned int pair_source_col = head_base + pair_head_col;
                const float pair_norm_weight =
                    pto_cuda_tensor_arg_f32(task, norm_slot, pair_head_col, 1.0f);
                const float paired =
                    task->a[row_base + pair_source_col] * scale *
                    pair_norm_weight;
                const unsigned int rope_index = first_half ?
                    head_col : head_col - half_head_dim;
                float cos_value = task->scalar_arg_count > 0U ?
                    task->scalar_args[0] : 1.0f;
                float sin_value = task->scalar_arg_count > 1U ?
                    task->scalar_args[1] : 0.0f;
                if (task->tensor_arg_count >= 4U &&
                    task->tensor_args[2] && task->tensor_args[3]) {
                    cos_value =
                        pto_cuda_tensor_arg_f32(task, 2U, rope_index, cos_value);
                    sin_value =
                        pto_cuda_tensor_arg_f32(task, 3U, rope_index, sin_value);
                }
                task->out[j] = first_half ?
                    normalized * cos_value - paired * sin_value :
                    normalized * cos_value + paired * sin_value;
            } else {
                task->out[j] = normalized;
            }
            if (!is_query_region && task->c) {
                const unsigned int logical_page = decode_position / kv_page_size;
                const unsigned int page_offset = decode_position % kv_page_size;
                const unsigned int physical_page = qk_norm_kv_page_table ?
                    qk_norm_kv_page_table[logical_page] : logical_page;
                const unsigned long long token_slot =
                    static_cast<unsigned long long>(physical_page) *
                        kv_page_size + page_offset;
                const unsigned long long qk_norm_kv_write_index =
                    kv_layer_base +
                    static_cast<unsigned long long>(row) * sequence_capacity *
                        kv_width + token_slot * kv_width + region_col;
                task->c[qk_norm_kv_write_index] = task->out[j];
            }
        }
    } else {
        __shared__ float partial[1024];
        const unsigned int fallback_row = 0U;
        const unsigned long long row_base =
            static_cast<unsigned long long>(fallback_row) * qk_norm_input_stride;
        float mean_square = 0.0f;
        for (unsigned long long k = threadIdx.x; k < task->inner;
             k += blockDim.x) {
            const float value = task->a[row_base + k];
            mean_square += value * value;
        }
        partial[threadIdx.x] = mean_square;
        __syncthreads();
        for (unsigned int reduction_stride = blockDim.x >> 1;
             reduction_stride > 0U; reduction_stride >>= 1) {
            if (threadIdx.x < reduction_stride) {
                partial[threadIdx.x] += partial[threadIdx.x + reduction_stride];
            }
            __syncthreads();
        }
        const float scale =
            rsqrtf(partial[0] / static_cast<float>(task->inner) + 0.000001f);
        for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
            const unsigned int col = static_cast<unsigned int>(j % task->cols);
            const float q_weight = pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
            const float k_weight = pto_cuda_tensor_arg_f32(task, 1U, col, 1.0f);
            const float normalized = task->a[row_base + col] * scale * 0.5f *
                (q_weight + k_weight);
            const unsigned int half_inner = task->inner >> 1U;
            if (half_inner > 0U) {
                const bool first_half = col < half_inner;
                const unsigned int pair_col = first_half ?
                    col + half_inner : col - half_inner;
                const float pair_q_weight =
                    pto_cuda_tensor_arg_f32(task, 0U, pair_col, 1.0f);
                const float pair_k_weight =
                    pto_cuda_tensor_arg_f32(task, 1U, pair_col, 1.0f);
                const float paired = task->a[row_base + pair_col] * scale *
                    0.5f * (pair_q_weight + pair_k_weight);
                const unsigned int rope_index = first_half ?
                    col : col - half_inner;
                float cos_value = task->scalar_arg_count > 0U ?
                    task->scalar_args[0] : 1.0f;
                float sin_value = task->scalar_arg_count > 1U ?
                    task->scalar_args[1] : 0.0f;
                if (task->tensor_arg_count >= 4U &&
                    task->tensor_args[2] && task->tensor_args[3]) {
                    cos_value =
                        pto_cuda_tensor_arg_f32(task, 2U, rope_index, cos_value);
                    sin_value =
                        pto_cuda_tensor_arg_f32(task, 3U, rope_index, sin_value);
                }
                task->out[j] = first_half ?
                    normalized * cos_value - paired * sin_value :
                    normalized * cos_value + paired * sin_value;
            } else {
                task->out[j] = normalized;
            }
        }
    }
} else {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const float q = pto_cuda_tensor_arg_f32(task, 0U, j & 3U, 1.0f);
        const float k = pto_cuda_tensor_arg_f32(task, 1U, j & 3U, 1.0f);
        task->out[j] = task->a[j] * 0.5f * (q + k);
    }
}
""",
        },
        {
            "callable": "qwen_attention_o",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "c", "d", "tensor_args", "scalar_args"],
            "consumes_roles": [
                "attention_state",
                "key_cache",
                "value_cache",
                "o_proj_weight",
                "kv_page_table",
            ],
            "threading": "block",
            "body": """
if (task->cols > 0U && task->inner > 0U && task->c && task->d) {
    __shared__ float attention_values[4096];
    const unsigned int row_count =
        static_cast<unsigned int>(task->n / task->cols);
    const unsigned int query_heads = task->rows > 0U ? task->rows : 1U;
    unsigned int head_dim = task->lda > 0U ?
        task->lda : (task->cols / query_heads);
    head_dim = head_dim > 0U ? head_dim : 1U;
    const unsigned int kv_heads = task->ldb > 0U ? task->ldb : query_heads;
    const unsigned int heads_per_kv =
        query_heads > kv_heads ? query_heads / kv_heads : 1U;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const unsigned int kv_window = task->inner;
    unsigned int kv_page_size =
        task->scalar0 > 0.0f ? static_cast<unsigned int>(task->scalar0) : kv_window;
    kv_page_size = kv_page_size > 0U ? kv_page_size : kv_window;
    const unsigned int sequence_capacity =
        task->b_batch_stride > 0U ? task->b_batch_stride : kv_page_size;
    const unsigned int cache_batch_size =
        task->out_batch_stride > 0U ? task->out_batch_stride : task->rows;
    const unsigned int kv_layer_index = task->scalar_arg_count > 3U ?
        static_cast<unsigned int>(task->scalar_args[3]) : 0U;
    const unsigned long long kv_layer_base =
        static_cast<unsigned long long>(kv_layer_index) * cache_batch_size *
        sequence_capacity * kv_heads * head_dim;
    unsigned int attention_tile =
        task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1) : 16U;
    attention_tile = attention_tile > 0U ? attention_tile : 16U;
    const unsigned int requested_projection_input_count =
        task->scalar_arg_count > 1U ?
        static_cast<unsigned int>(task->scalar_args[1]) : task->cols;
    const unsigned int projection_input_count =
        requested_projection_input_count < task->cols ?
        requested_projection_input_count : task->cols;
    const unsigned int cached_projection_input_count =
        projection_input_count < 4096U ? projection_input_count : 4096U;
    const unsigned int projection_stride =
        task->ldc > 0U ? task->ldc : task->cols;
    const unsigned int *kv_page_table =
        task->tensor_arg_count > 1U && task->tensor_args[1] ?
        reinterpret_cast<const unsigned int *>(task->tensor_args[1]) : nullptr;
    if (task->tensor_arg_count > 0U && task->tensor_args[0]) {
        for (unsigned int row = 0U; row < row_count; ++row) {
            const unsigned int input_stride =
                task->a_batch_stride > 0U ? task->a_batch_stride : task->cols;
            const unsigned long long row_base =
                static_cast<unsigned long long>(row) * input_stride;
            const unsigned long long kv_read_base =
                kv_layer_base +
                static_cast<unsigned long long>(row) * sequence_capacity *
                kv_heads * head_dim;
            for (unsigned int projection_col = threadIdx.x;
                 projection_col < cached_projection_input_count;
                 projection_col += blockDim.x) {
                const unsigned int projection_query_head =
                    projection_col / head_dim;
                const unsigned int projection_head_col =
                    projection_col % head_dim;
                const unsigned int projection_query_base =
                    projection_query_head * head_dim;
                const unsigned int projection_mapped_kv_head =
                    projection_query_head / heads_per_kv;
                const unsigned int projection_kv_head =
                    projection_mapped_kv_head < kv_heads ?
                    projection_mapped_kv_head : kv_heads - 1U;
                float projection_max_score = -3.4028234663852886e+38f;
                for (unsigned int tile_begin = 0U; tile_begin < kv_window;
                     tile_begin += attention_tile) {
                    const unsigned int tile_end =
                        tile_begin + attention_tile < kv_window ?
                        tile_begin + attention_tile : kv_window;
                    for (unsigned int step = tile_begin; step < tile_end; ++step) {
                        const unsigned int logical_page = step / kv_page_size;
                        const unsigned int page_offset = step % kv_page_size;
                        const unsigned int physical_page = kv_page_table ?
                            kv_page_table[logical_page] : logical_page;
                        float score = 0.0f;
                        for (unsigned int dim = 0U; dim < head_dim; ++dim) {
                            const unsigned long long kv_index =
                                kv_read_base +
                                static_cast<unsigned long long>(physical_page) *
                                    kv_page_size * kv_heads * head_dim +
                                static_cast<unsigned long long>(page_offset) *
                                    kv_heads * head_dim +
                                static_cast<unsigned long long>(
                                    projection_kv_head) * head_dim + dim;
                            score += task->a[
                                row_base + projection_query_base + dim] *
                                task->c[kv_index];
                        }
                        score *= attention_scale;
                        projection_max_score = score > projection_max_score ?
                            score : projection_max_score;
                    }
                }
                float projection_weighted_value = 0.0f;
                float projection_normalizer = 0.0f;
                for (unsigned int tile_begin = 0U; tile_begin < kv_window;
                     tile_begin += attention_tile) {
                    const unsigned int tile_end =
                        tile_begin + attention_tile < kv_window ?
                        tile_begin + attention_tile : kv_window;
                    for (unsigned int step = tile_begin; step < tile_end; ++step) {
                        const unsigned int logical_page = step / kv_page_size;
                        const unsigned int page_offset = step % kv_page_size;
                        const unsigned int physical_page = kv_page_table ?
                            kv_page_table[logical_page] : logical_page;
                        float score = 0.0f;
                        for (unsigned int dim = 0U; dim < head_dim; ++dim) {
                            const unsigned long long score_kv_index =
                                kv_read_base +
                                static_cast<unsigned long long>(physical_page) *
                                    kv_page_size * kv_heads * head_dim +
                                static_cast<unsigned long long>(page_offset) *
                                    kv_heads * head_dim +
                                static_cast<unsigned long long>(
                                    projection_kv_head) * head_dim + dim;
                            score += task->a[
                                row_base + projection_query_base + dim] *
                                task->c[score_kv_index];
                        }
                        score *= attention_scale;
                        const unsigned long long value_kv_index =
                            kv_read_base +
                            static_cast<unsigned long long>(physical_page) *
                                kv_page_size * kv_heads * head_dim +
                            static_cast<unsigned long long>(page_offset) *
                                kv_heads * head_dim +
                            static_cast<unsigned long long>(projection_kv_head) *
                                head_dim + projection_head_col;
                        const float weight =
                            expf(score - projection_max_score);
                        projection_weighted_value +=
                            weight * task->d[value_kv_index];
                        projection_normalizer += weight;
                    }
                }
                attention_values[projection_col] =
                    projection_normalizer > 0.0f ?
                    projection_weighted_value / projection_normalizer : 0.0f;
            }
            __syncthreads();
            for (unsigned int col = threadIdx.x; col < task->cols;
                 col += blockDim.x) {
                float projected_attention = 0.0f;
                for (unsigned int projection_col = 0U;
                     projection_col < cached_projection_input_count;
                     ++projection_col) {
                    const unsigned long long o_weight_index =
                        static_cast<unsigned long long>(col) *
                        projection_stride + projection_col;
                    const float o_weight =
                        pto_cuda_tensor_arg_f32(
                            task, 0U, o_weight_index, 0.0f);
                    projected_attention +=
                        attention_values[projection_col] * o_weight;
                }
                task->out[
                    static_cast<unsigned long long>(row) * task->cols + col] =
                    projected_attention;
            }
            __syncthreads();
        }
    } else {
        for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
            const unsigned int row = static_cast<unsigned int>(j / task->cols);
            const unsigned int col = static_cast<unsigned int>(j % task->cols);
            const unsigned int query_head = col / head_dim;
            const unsigned int head_col = col % head_dim;
            const unsigned int mapped_kv_head = query_head / heads_per_kv;
            const unsigned int kv_head =
                mapped_kv_head < kv_heads ? mapped_kv_head : kv_heads - 1U;
            const unsigned int input_stride =
                task->a_batch_stride > 0U ? task->a_batch_stride : task->cols;
            const unsigned long long row_base =
                static_cast<unsigned long long>(row) * input_stride;
            const unsigned int query_base = query_head * head_dim;
            const unsigned long long kv_read_base =
                kv_layer_base +
                static_cast<unsigned long long>(row) * sequence_capacity *
                kv_heads * head_dim;
            float max_score = -3.4028234663852886e+38f;
            for (unsigned int tile_begin = 0U; tile_begin < kv_window;
                 tile_begin += attention_tile) {
                const unsigned int tile_end =
                    tile_begin + attention_tile < kv_window ?
                    tile_begin + attention_tile : kv_window;
                for (unsigned int step = tile_begin; step < tile_end; ++step) {
                    const unsigned int logical_page = step / kv_page_size;
                    const unsigned int page_offset = step % kv_page_size;
                    const unsigned int physical_page = kv_page_table ?
                        kv_page_table[logical_page] : logical_page;
                    float score = 0.0f;
                    for (unsigned int dim = 0U; dim < head_dim; ++dim) {
                        const unsigned long long kv_index =
                            kv_read_base +
                            static_cast<unsigned long long>(physical_page) *
                                kv_page_size * kv_heads * head_dim +
                            static_cast<unsigned long long>(page_offset) *
                                kv_heads * head_dim +
                            static_cast<unsigned long long>(kv_head) *
                                head_dim + dim;
                        score += task->a[row_base + query_base + dim] *
                            task->c[kv_index];
                    }
                    score *= attention_scale;
                    max_score = score > max_score ? score : max_score;
                }
            }
            float weighted_value = 0.0f;
            float normalizer = 0.0f;
            for (unsigned int tile_begin = 0U; tile_begin < kv_window;
                 tile_begin += attention_tile) {
                const unsigned int tile_end =
                    tile_begin + attention_tile < kv_window ?
                    tile_begin + attention_tile : kv_window;
                for (unsigned int step = tile_begin; step < tile_end; ++step) {
                    const unsigned int logical_page = step / kv_page_size;
                    const unsigned int page_offset = step % kv_page_size;
                    const unsigned int physical_page = kv_page_table ?
                        kv_page_table[logical_page] : logical_page;
                    float score = 0.0f;
                    for (unsigned int dim = 0U; dim < head_dim; ++dim) {
                        const unsigned long long score_kv_index =
                            kv_read_base +
                            static_cast<unsigned long long>(physical_page) *
                                kv_page_size * kv_heads * head_dim +
                            static_cast<unsigned long long>(page_offset) *
                                kv_heads * head_dim +
                            static_cast<unsigned long long>(kv_head) *
                                head_dim + dim;
                        score += task->a[row_base + query_base + dim] *
                            task->c[score_kv_index];
                    }
                    score *= attention_scale;
                    const unsigned long long value_kv_index =
                        kv_read_base +
                        static_cast<unsigned long long>(physical_page) *
                            kv_page_size * kv_heads * head_dim +
                        static_cast<unsigned long long>(page_offset) *
                            kv_heads * head_dim +
                        static_cast<unsigned long long>(kv_head) * head_dim +
                            head_col;
                    const float weight = expf(score - max_score);
                    weighted_value += weight * task->d[value_kv_index];
                    normalizer += weight;
                }
            }
            task->out[j] =
                normalizer > 0.0f ? weighted_value / normalizer : 0.0f;
        }
    }
} else if (task->cols > 0U && task->inner > 0U &&
    task->tensor_arg_count > 0U && task->tensor_args[0]) {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int row = static_cast<unsigned int>(j / task->cols);
        const unsigned int col = static_cast<unsigned int>(j % task->cols);
        task->out[j] = pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f);
    }
} else {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const float weight = pto_cuda_tensor_arg_f32(task, 0U, j & 3U, 0.0f);
        task->out[j] = task->a[j] + weight;
    }
}
""",
        },
        {
            "callable": "qwen_rmsnorm_post_attention",
            "phase": "per_layer_decode",
            "threading": "block",
            "consumes_fields": ["a", "b", "out", "tensor_args"],
            "consumes_roles": [
                "attention_output",
                "attention_residual",
                "post_attention_layernorm_weight",
            ],
            "body": """
if (task->scalar_arg_count > 1 && task->scalar_args[0] == 1.0f &&
    task->scalar_args[1] == 0.0f && task->cols > 0U) {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int row = static_cast<unsigned int>(j / task->cols);
        const unsigned int col = static_cast<unsigned int>(j % task->cols);
        const unsigned int input_stride =
            task->lda > 0U ? task->lda : task->cols;
        const unsigned long long row_base =
            static_cast<unsigned long long>(row) * input_stride;
        float mean_square = 0.0f;
        for (unsigned int k = 0U; k < task->cols; ++k) {
            const float residual_value =
                task->b ? task->b[row_base + k] : 0.0f;
            const float value = task->a[row_base + k] + residual_value;
            mean_square += value * value;
        }
        const float scale =
            rsqrtf(mean_square / static_cast<float>(task->cols) + 0.000001f);
        const float residual_value = task->b ? task->b[row_base + col] : 0.0f;
        const float value = task->a[row_base + col] + residual_value;
        const float weight = pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
        task->out[j] = value * scale * weight;
    }
} else {
    const float external_scale =
        task->scalar_arg_count > 1 ? task->scalar_args[1] : 1.0f;
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int col = static_cast<unsigned int>(
            task->cols > 0U ? j % task->cols : j);
        const float residual_value = task->b ? task->b[j] : 0.0f;
        const float value = task->a[j] + residual_value;
        task->out[j] = value * external_scale *
            pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
    }
}
""",
        },
        {
            "callable": "qwen_mlp_gate_up",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "gate_proj_weight", "up_proj_weight"],
            "body": """
if (task->cols > 0U && task->inner > 0U && task->tensor_arg_count >= 2U &&
    task->tensor_args[0] && task->tensor_args[1]) {
    const unsigned int row = static_cast<unsigned int>(i / task->cols);
    const unsigned int col = static_cast<unsigned int>(i % task->cols);
    const unsigned int requested_active_projection_cols =
        task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1) :
        task->cols;
    const unsigned int active_projection_cols =
        requested_active_projection_cols < task->cols ?
        requested_active_projection_cols : task->cols;
    if (col < active_projection_cols) {
        const float gate_value =
            pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f);
        const float up_value =
            pto_cuda_linear_arg_f32(task, 1U, row, col, 0.0f);
        task->out[i] = pto_cuda_silu(gate_value) * up_value;
    } else {
        task->out[i] = 0.0f;
    }
} else {
    const float gate_value =
        pto_cuda_tensor_arg_f32(task, 0U, i & 3U, task->a[i]);
    const float up_value =
        pto_cuda_tensor_arg_f32(task, 1U, i & 3U, task->a[i]);
    task->out[i] = pto_cuda_silu(gate_value) * up_value;
}
""",
        },
        {
            "callable": "qwen_mlp_down",
            "phase": "per_layer_decode",
            "consumes_fields": ["a", "b", "out", "tensor_args"],
            "consumes_roles": ["mlp_state", "mlp_residual", "down_proj_weight"],
            "body": """
if (task->cols > 0U && task->inner > 0U && task->tensor_arg_count > 0U &&
    task->tensor_args[0]) {
    const unsigned int row = static_cast<unsigned int>(i / task->cols);
    const unsigned int col = static_cast<unsigned int>(i % task->cols);
    const unsigned int requested_active_projection_cols =
        task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1) :
        task->cols;
    const unsigned int active_projection_cols =
        requested_active_projection_cols < task->cols ?
        requested_active_projection_cols : task->cols;
    const float residual_value = task->b ? task->b[i] : 0.0f;
    const float projected_down = col < active_projection_cols ?
        pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f) : 0.0f;
    task->out[i] = projected_down + residual_value;
} else {
    const float down = pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    const float residual_value = task->b ? task->b[i] : task->a[i];
    task->out[i] = residual_value + down;
}
""",
        },
        {
            "callable": "qwen_final_norm",
            "phase": "per_token_decode",
            "threading": "block",
            "consumes_fields": ["a", "out", "tensor_args"],
            "consumes_roles": ["hidden_state", "final_norm_weight"],
            "body": """
if (task->scalar_arg_count > 1 && task->scalar_args[0] == 1.0f &&
    task->scalar_args[1] == 0.0f && task->cols > 0U) {
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int row = static_cast<unsigned int>(j / task->cols);
        const unsigned int col = static_cast<unsigned int>(j % task->cols);
        const unsigned int input_stride =
            task->lda > 0U ? task->lda : task->cols;
        const unsigned long long row_base =
            static_cast<unsigned long long>(row) * input_stride;
        float mean_square = 0.0f;
        for (unsigned int k = 0U; k < task->cols; ++k) {
            const float value = task->a[row_base + k];
            mean_square += value * value;
        }
        const float scale =
            rsqrtf(mean_square / static_cast<float>(task->cols) + 0.000001f);
        const float weight = pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
        task->out[j] = task->a[row_base + col] * scale * weight;
    }
} else {
    const float external_scale =
        task->scalar_arg_count > 1 ? task->scalar_args[1] : 1.0f;
    for (unsigned long long j = threadIdx.x; j < task->n; j += blockDim.x) {
        const unsigned int col = static_cast<unsigned int>(
            task->cols > 0U ? j % task->cols : j);
        task->out[j] = task->a[j] * external_scale *
            pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f);
    }
}
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


def source_preview(source: str, callables: list[str], *, window_lines: int = 64) -> str:
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
                **(
                    {"decode_feedback_scope": spec["decode_feedback_scope"]}
                    if "decode_feedback_scope" in spec
                    else {}
                ),
            }
            for function, spec in zip(functions, specs, strict=True)
        ],
        "coverage": {
            "token_fields": ["a", "b", "out"],
            "kv_fields": ["c", "d"],
            "kv_write_policy": "slot_mapped_kv_cache_writeback_ready",
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
                    "qwen_attention_o",
                    "qwen_mlp_gate_up",
                    "qwen_logits",
                ],
            ),
        },
        "numeric_oracle": build_numeric_oracle(callables),
        "qwen_unit_math_oracle": build_qwen_unit_math_oracle(),
        "qwen_decode_attention_oracle": build_qwen_decode_attention_oracle(),
        "qwen_tensor_tile_contract": build_qwen_tensor_tile_contract(),
        "qwen_kernel_source_map": build_kernel_source_map(),
        "implemented_contracts": [
            "generated_qwen_kernel_bodies",
            "controlled_proxy_numeric_oracle",
            "qwen_unit_math_oracle",
            "qwen_tensor_tile_source_contract",
            "qwen_kernel_source_map",
            "qwen_unit_math_source_coverage",
            "qwen_embedding_shape_lookup_source",
            "qwen_input_rmsnorm_hidden_weight_source",
            "qwen_rowwise_rmsnorm_batch_source",
            "qwen_shape_field_linear_projection_source",
            "qwen_shape_field_qk_rmsnorm_source",
            "qwen_post_attention_norm_full_rmsnorm_source",
            "qwen_post_attention_residual_rmsnorm_source",
            "qwen_qk_norm_block_rmsnorm_rope_source",
            "qwen_qk_norm_rotate_half_rope_source",
            "qwen_qk_norm_separate_qk_regions_source",
            "qwen_qk_norm_normalized_k_cache_writeback_source",
            "qwen_qk_norm_paged_k_cache_writeback_source",
            "qwen_qk_norm_batch_row_index_source",
            "qwen_qk_norm_qkv_input_stride_source",
            "qwen_final_norm_full_rmsnorm_source",
            "qwen_shape_field_qk_rope_source",
            "qwen_bounded_decode_attention_reduction_source",
            "qwen_decode_attention_dot_product_source",
            "qwen_decode_attention_head_dim_scale_source",
            "qwen_attention_o_batch_local_kv_read_source",
            "qwen_attention_o_qk_norm_input_stride_source",
            "qwen_attention_o_bounded_projection_source",
            "qwen_attention_o_cached_projection_source",
            "qwen_mlp_down_residual_add_source",
            "qwen_gqa_decode_attention_head_grouping_source",
            "qwen_paged_kv_attention_index_source",
            "qwen_layer_partitioned_kv_cache_source",
            "qwen_tiled_decode_attention_softmax_source",
            "qwen_logits_full_vocab_argmax_source",
            "qwen_logits_tiled_vocab_projection_source",
            "qwen_kernel_token_field_consumption",
            "qwen_kernel_kv_field_consumption",
            "qwen_slot_mapped_kv_cache_writeback_source",
            "qwen_kernel_kv_cache_writeback_field_contract",
            "qwen_kernel_weight_tensor_arg_consumption",
            "qwen_logits_device_sampled_token_feedback_source",
            "qwen_decode_feedback_prompt_ring_source",
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
