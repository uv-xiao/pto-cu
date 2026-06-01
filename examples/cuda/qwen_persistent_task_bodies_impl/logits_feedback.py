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
const float *lm_head = task->tensor_args[0];
if (task->scalar_arg_count > 1 && lm_head) {
    const unsigned long long hidden_elements =
        static_cast<unsigned long long>(task->scalar_args[1]);
    const unsigned long long hidden_index = i % max(1ULL, hidden_elements);
    task->out[i] = task->a[hidden_index] * lm_head[i & 3U];
    if (task->scalar_arg_count > 3 && task->tensor_args[2] &&
        task->tensor_args[3] && i == 0) {
        unsigned int best_token = 0;
        float best_logit = task->a[0] * lm_head[0];
        for (unsigned int token = 1; token < 4; ++token) {
            const unsigned long long candidate_index =
                token % max(1ULL, hidden_elements);
            const float candidate =
                task->a[candidate_index] * lm_head[token & 3U];
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
    const float logit = task->a[i] + (lm_head ? lm_head[i & 3U] : 0.0f);
    task->out[i] = logit;
}
""",
    }
