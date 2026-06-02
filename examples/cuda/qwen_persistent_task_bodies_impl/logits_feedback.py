"""Qwen logits task-body spec with diagnostic device token feedback."""

from __future__ import annotations

from typing import Any


def qwen_logits_spec() -> dict[str, Any]:
    return {
        "callable": "qwen_logits",
        "phase": "per_token_decode",
        "consumes_fields": ["a", "out", "tensor_args", "scalar_args"],
        "consumes_roles": [
            "hidden_state",
            "lm_head_weight",
            "input_ids_feedback",
            "output_ids_feedback",
            "decode_step_index",
        ],
        "body": """
const bool has_lm_head = task->tensor_arg_count > 0U && task->tensor_args[0];
if (task->cols > 0U && task->inner > 0U && has_lm_head) {
    const unsigned int row = static_cast<unsigned int>(i / task->cols);
    const unsigned int col = static_cast<unsigned int>(i % task->cols);
    const unsigned int hidden_width = task->inner;
    const unsigned int hidden_stride = task->lda > 0U ? task->lda : hidden_width;
    const unsigned int weight_stride = task->ldb > 0U ? task->ldb : hidden_width;
    const unsigned int requested_logits_tile =
        task->scalar0 > 0.0f ? static_cast<unsigned int>(task->scalar0) : 256U;
    const unsigned int logits_tile =
        requested_logits_tile > 0U ? requested_logits_tile : 256U;
    float acc = 0.0f;
    for (unsigned int tile_begin = 0U; tile_begin < hidden_width;
         tile_begin += logits_tile) {
        const unsigned int tile_end =
            tile_begin + logits_tile < hidden_width ?
            tile_begin + logits_tile : hidden_width;
        for (unsigned int k = tile_begin; k < tile_end; ++k) {
            const unsigned long long a_index =
                static_cast<unsigned long long>(row) * hidden_stride + k;
            const unsigned long long weight_index =
                static_cast<unsigned long long>(col) * weight_stride + k;
            acc += task->a[a_index] *
                pto_cuda_tensor_arg_f32(task, 0U, weight_index, 0.0f);
        }
    }
    task->out[i] = acc;
    if (task->scalar_arg_count > 3 && task->tensor_args[2] &&
        task->tensor_args[3] && i == 0) {
        unsigned int best_token = 0;
        float best_logit = task->out[0];
        for (unsigned int token = 1; token < task->cols; ++token) {
            float candidate = 0.0f;
            for (unsigned int tile_begin = 0U; tile_begin < hidden_width;
                 tile_begin += logits_tile) {
                const unsigned int tile_end =
                    tile_begin + logits_tile < hidden_width ?
                    tile_begin + logits_tile : hidden_width;
                for (unsigned int k = tile_begin; k < tile_end; ++k) {
                    const unsigned long long a_index =
                        static_cast<unsigned long long>(row) * hidden_stride + k;
                    const unsigned long long weight_index =
                        static_cast<unsigned long long>(token) * weight_stride + k;
                    candidate += task->a[a_index] *
                        pto_cuda_tensor_arg_f32(task, 0U, weight_index, 0.0f);
                }
            }
            if (candidate > best_logit) {
                best_logit = candidate;
                best_token = token;
            }
        }
        unsigned int *input_ids =
            const_cast<unsigned int *>(
                reinterpret_cast<const unsigned int *>(task->tensor_args[2]));
        unsigned int *output_ids =
            const_cast<unsigned int *>(
                reinterpret_cast<const unsigned int *>(task->tensor_args[3]));
        const unsigned long long decode_step =
            static_cast<unsigned long long>(task->scalar_args[3]);
        output_ids[decode_step] = best_token;
        input_ids[0] = best_token;
    }
} else if (task->scalar_arg_count > 1 && has_lm_head) {
    const unsigned long long hidden_elements =
        static_cast<unsigned long long>(task->scalar_args[1]);
    const unsigned long long hidden_index = i % max(1ULL, hidden_elements);
    const float lm_head =
        pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    task->out[i] = task->a[hidden_index] * lm_head;
    if (task->scalar_arg_count > 3 && task->tensor_args[2] &&
        task->tensor_args[3] && i == 0) {
        unsigned int best_token = 0;
        float best_logit =
            task->a[0] * pto_cuda_tensor_arg_f32(task, 0U, 0U, 0.0f);
        for (unsigned int token = 1; token < 4; ++token) {
            const unsigned long long candidate_index =
                token % max(1ULL, hidden_elements);
            const float candidate =
                task->a[candidate_index] *
                pto_cuda_tensor_arg_f32(task, 0U, token & 3U, 0.0f);
            if (candidate > best_logit) {
                best_logit = candidate;
                best_token = token;
            }
        }
        unsigned int *input_ids =
            const_cast<unsigned int *>(
                reinterpret_cast<const unsigned int *>(task->tensor_args[2]));
        unsigned int *output_ids =
            const_cast<unsigned int *>(
                reinterpret_cast<const unsigned int *>(task->tensor_args[3]));
        const unsigned long long decode_step =
            static_cast<unsigned long long>(task->scalar_args[3]);
        output_ids[decode_step] = best_token;
        input_ids[0] = best_token;
    }
} else {
    const float logit =
        task->a[i] + pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
    task->out[i] = logit;
}
""",
    }
