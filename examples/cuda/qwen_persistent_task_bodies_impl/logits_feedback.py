"""Qwen logits task-body spec with diagnostic device token feedback."""

from __future__ import annotations

from typing import Any


def qwen_logits_spec() -> dict[str, Any]:
    return {
        "callable": "qwen_logits",
        "phase": "per_token_decode",
        "decode_feedback_scope": "single_sequence_row0_greedy_argmax",
        "consumes_fields": ["a", "out", "tensor_args", "scalar_args"],
        "consumes_roles": [
            "hidden_state",
            "lm_head_weight",
            "input_ids_feedback",
            "output_ids_feedback",
            "decode_step_index",
        ],
        "threading": "block",
        "body": """
const bool has_lm_head = task->tensor_arg_count > 0U && task->tensor_args[0];
if (task->cols > 0U && task->inner > 0U && has_lm_head) {
    const unsigned int hidden_width = task->inner;
    const unsigned int hidden_stride = task->lda > 0U ? task->lda : hidden_width;
    const unsigned int weight_stride = task->ldb > 0U ? task->ldb : hidden_width;
    const unsigned int requested_logits_tile =
        task->scalar0 > 0.0f ? static_cast<unsigned int>(task->scalar0) : 256U;
    const unsigned int logits_tile =
        requested_logits_tile > 0U ? requested_logits_tile : 256U;
    const unsigned int requested_active_logits_cols =
        task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1) :
        task->cols;
    const unsigned int active_logits_cols =
        requested_active_logits_cols < task->cols ?
        requested_active_logits_cols : task->cols;
    const unsigned long long row_count =
        task->n / static_cast<unsigned long long>(task->cols);
    const unsigned long long active_logits_elements =
        row_count * static_cast<unsigned long long>(active_logits_cols);
    for (unsigned long long i = threadIdx.x; i < active_logits_elements;
         i += blockDim.x) {
        const unsigned int row =
            static_cast<unsigned int>(i / active_logits_cols);
        const unsigned int col =
            static_cast<unsigned int>(i % active_logits_cols);
        const unsigned long long output_index =
            static_cast<unsigned long long>(row) * task->cols + col;
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
        task->out[output_index] = pto_cuda_round_to_bf16_f32(acc);
    }
    __syncthreads();
    if (task->scalar_arg_count > 3 && task->tensor_args[2] &&
        task->tensor_args[3]) {
        __shared__ float logits_best_values[1024];
        __shared__ unsigned int logits_best_tokens[1024];
        float local_best_logit = -3.4028234663852886e38f;
        unsigned int local_best_token = 0U;
        for (unsigned int token = threadIdx.x; token < active_logits_cols;
             token += blockDim.x) {
            const float candidate = task->out[token];
            if (candidate > local_best_logit) {
                local_best_logit = candidate;
                local_best_token = token;
            }
        }
        logits_best_values[threadIdx.x] = local_best_logit;
        logits_best_tokens[threadIdx.x] = local_best_token;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0U; stride >>= 1) {
            if (threadIdx.x < stride &&
                logits_best_values[threadIdx.x + stride] >
                logits_best_values[threadIdx.x]) {
                logits_best_values[threadIdx.x] =
                    logits_best_values[threadIdx.x + stride];
                logits_best_tokens[threadIdx.x] =
                    logits_best_tokens[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            unsigned int *input_ids =
                const_cast<unsigned int *>(
                    reinterpret_cast<const unsigned int *>(task->tensor_args[2]));
            unsigned int *output_ids =
                const_cast<unsigned int *>(
                    reinterpret_cast<const unsigned int *>(task->tensor_args[3]));
            const unsigned long long decode_step =
                static_cast<unsigned long long>(task->scalar_args[3]);
            output_ids[decode_step] = logits_best_tokens[0];
            const unsigned long long next_input_index =
                static_cast<unsigned long long>(task->scalar_args[2]) + 1ULL;
            const unsigned long long prompt_stride =
                task->a_batch_stride > 0ULL ? task->a_batch_stride : 1ULL;
            const unsigned long long feedback_input_index =
                prompt_stride > 0ULL ? next_input_index % prompt_stride : 0ULL;
            input_ids[feedback_input_index] = logits_best_tokens[0];
        }
    }
} else if (task->scalar_arg_count > 1 && has_lm_head) {
    const unsigned long long hidden_elements =
        static_cast<unsigned long long>(task->scalar_args[1]);
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const unsigned long long hidden_index = i % max(1ULL, hidden_elements);
        const float lm_head =
            pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
        task->out[i] = task->a[hidden_index] * lm_head;
    }
    __syncthreads();
    if (task->scalar_arg_count > 3 && task->tensor_args[2] &&
        task->tensor_args[3] && threadIdx.x == 0) {
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
        const unsigned long long next_input_index =
            static_cast<unsigned long long>(task->scalar_args[2]) + 1ULL;
        const unsigned long long prompt_stride =
            task->a_batch_stride > 0ULL ? task->a_batch_stride : 1ULL;
        const unsigned long long feedback_input_index =
            prompt_stride > 0ULL ? next_input_index % prompt_stride : 0ULL;
        input_ids[feedback_input_index] = best_token;
    }
} else {
    for (unsigned long long i = threadIdx.x; i < task->n; i += blockDim.x) {
        const float logit =
            task->a[i] + pto_cuda_tensor_arg_f32(task, 0U, i & 3U, 0.0f);
        task->out[i] = logit;
    }
}
""",
    }
