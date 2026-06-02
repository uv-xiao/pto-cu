#!/usr/bin/env python3
"""Emit Qwen persistent-device task body source-generation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_persistent_task_bodies_impl.lifecycle import (  # noqa: E402
    build_task_body_manifest,
    repo_relative,
    write_json,
    write_source,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_persistent_task_bodies",
    "generated_qwen_kernel_bodies",
    "controlled_proxy_numeric_oracle",
    "qwen_unit_math_oracle",
    "qwen_tensor_tile_source_contract",
    "qwen_unit_math_source_coverage",
    "qwen_kernel_token_field_consumption",
    "qwen_embedding_shape_lookup_source",
    "qwen_shape_field_qk_rmsnorm_source",
    "qwen_post_attention_norm_full_rmsnorm_source",
    "qwen_post_attention_residual_rmsnorm_source",
    "qwen_qk_norm_block_rmsnorm_rope_source",
    "qwen_qk_norm_separate_qk_regions_source",
    "qwen_qk_norm_normalized_k_cache_writeback_source",
    "qwen_final_norm_full_rmsnorm_source",
    "qwen_shape_field_qk_rope_source",
    "qwen_bounded_decode_attention_reduction_source",
    "qwen_attention_o_bounded_projection_source",
    "qwen_gqa_decode_attention_head_grouping_source",
    "qwen_paged_kv_attention_index_source",
    "qwen_tiled_decode_attention_softmax_source",
    "qwen_logits_full_vocab_argmax_source",
    "qwen_logits_tiled_vocab_projection_source",
    "qwen_kernel_kv_field_consumption",
    "qwen_slot_mapped_kv_cache_writeback_source",
    "qwen_kernel_weight_tensor_arg_consumption",
    "qwen_logits_device_sampled_token_feedback_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-hidden-layers", type=int, default=36)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_task_body_manifest(num_hidden_layers=args.num_hidden_layers)
    if args.output_source:
        payload["rendered_source"]["artifact"] = write_source(args.output_source)
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
