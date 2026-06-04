import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_task_bodies_module():
    script_path = ROOT / "examples" / "cuda" / "qwen_persistent_task_bodies.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_persistent_task_bodies",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_body_manifest_tracks_qwen_unit_math_oracle():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)

    oracle = manifest["qwen_unit_math_oracle"]
    assert oracle["status"] == "qwen_unit_math_oracle_ready"
    assert oracle["scope"] == "single_token_hidden4_reference"
    assert oracle["hidden_size"] == 4
    assert oracle["checked_equations"] == [
        "rmsnorm",
        "linear_projection",
        "single_token_attention_cache_writeback",
        "silu",
        "swiglu",
        "logits_linear",
    ]
    assert oracle["steps"]["rmsnorm_input"] == [
        0.365148,
        -0.803326,
        1.150217,
        -2.19089,
    ]
    assert oracle["steps"]["attention_context"] == [
        0.146059,
        -0.240998,
        0.345065,
        -0.438178,
    ]
    assert oracle["steps"]["mlp_swiglu"] == [
        0.054983,
        -0.05402,
        0.060482,
        -0.063023,
    ]
    assert oracle["steps"]["logits"] == [
        0.186944,
        -0.237688,
        0.302409,
        -0.378139,
    ]
    proxy_oracle = manifest["numeric_oracle"]
    proxy_mlp = next(
        item
        for item in proxy_oracle["sample_outputs"]
        if item["callable"] == "qwen_mlp_gate_up"
    )
    assert proxy_mlp["expected_out"] == [
        0.365529,
        2.642391,
        7.144306,
        13.748193,
    ]
    assert "qwen_unit_math_oracle" in manifest["implemented_contracts"]
    assert "cuda_live_qwen_unit_math_execution" in manifest[
        "remaining_runtime_gaps"
    ]


def test_logits_task_body_declares_single_sequence_feedback_scope():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    logits = next(
        item for item in manifest["task_bodies"] if item["callable"] == "qwen_logits"
    )

    assert logits["decode_feedback_scope"] == "single_sequence_row0_greedy_argmax"
    assert "qwen_logits_device_sampled_token_feedback_source" in manifest[
        "implemented_contracts"
    ]


def test_generated_qwen_decode_feedback_wraps_prompt_ring_source():
    module = load_task_bodies_module()
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    full_source = render_persistent_dag_source(task_functions())

    assert "qwen_decode_feedback_prompt_ring_source" in manifest[
        "implemented_contracts"
    ]
    assert "const unsigned int token_position = prompt_stride > 0U ?" in full_source
    assert "requested_token_position % prompt_stride : 0U;" in full_source
    assert "const unsigned long long feedback_input_index =" in full_source
    assert "next_input_index % prompt_stride : 0ULL;" in full_source
    assert "input_ids[feedback_input_index] = logits_best_tokens[0];" in full_source


def test_rmsnorm_diagnostic_fallback_keeps_decode_position_scale_neutral():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    assert "task->scalar_args[1] != 0.0f ? task->scalar_args[1] : 1.0f" in (
        full_source
    )


def test_qk_rmsnorm_matches_hf_bf16_output_boundary():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    assert "__device__ float pto_cuda_round_to_bf16_f32(float value)" in full_source
    compact_source = " ".join(full_source.split())
    assert (
        "const float normalized = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_round_to_bf16_f32( "
        "task->a[row_base + source_col] * scale) * norm_weight);"
    ) in compact_source
    assert (
        "const float paired = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_round_to_bf16_f32( "
        "task->a[row_base + pair_source_col] * scale) * "
        "pair_norm_weight);"
    ) in compact_source
    assert "cos_value = pto_cuda_round_to_bf16_f32(cos_value);" in full_source
    assert "sin_value = pto_cuda_round_to_bf16_f32(sin_value);" in full_source
    assert "task->out[j] = pto_cuda_round_to_bf16_f32(first_half ?" in full_source


def test_qkv_projection_matches_hf_bf16_output_boundary():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    compact_source = " ".join(full_source.split())
    assert (
        "projected = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f));"
    ) in compact_source
    assert (
        "projected = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_linear_arg_f32(task, 1U, row, kv_col, 0.0f));"
    ) in compact_source
    assert (
        "projected = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_linear_arg_f32(task, 2U, row, kv_col, 0.0f));"
    ) in compact_source
    assert "task->c[kv_write_index] = projected;" in full_source
    assert "task->d[kv_write_index] = projected;" in full_source


def test_qwen_rmsnorm_outputs_match_hf_bf16_boundary():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    compact_source = " ".join(full_source.split())
    assert (
        "task->out[j] = pto_cuda_round_to_bf16_f32( "
        "task->a[row_base + col] * scale * norm_weight);"
    ) in compact_source
    assert (
        "task->out[j] = pto_cuda_round_to_bf16_f32("
        "value * scale * weight);"
    ) in compact_source
    assert (
        "task->out[j] = pto_cuda_round_to_bf16_f32( "
        "task->a[row_base + col] * scale * weight);"
    ) in compact_source


def test_qwen_residual_stream_matches_hf_bf16_boundaries():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    compact_source = " ".join(full_source.split())
    assert (
        "task->out[output_index] = "
        "pto_cuda_round_to_bf16_f32(projected_attention);"
    ) in compact_source
    assert (
        "const float value = pto_cuda_round_to_bf16_f32( "
        "task->a[row_base + col] + residual_value);"
    ) in compact_source
    assert (
        "task->out[i] = pto_cuda_round_to_bf16_f32("
        "projected_down + residual_value);"
    ) in compact_source


def test_qwen_logits_match_hf_bf16_output_boundary():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    full_source = render_persistent_dag_source(task_functions())

    compact_source = " ".join(full_source.split())
    assert (
        "task->out[output_index] = pto_cuda_round_to_bf16_f32(acc);"
    ) in compact_source


def test_generated_source_contains_qwen_unit_math_kernels():
    module = load_task_bodies_module()
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    source = manifest["rendered_source"]["preview"]
    full_source = render_persistent_dag_source(task_functions())
    compact_source = " ".join(full_source.split())
    source_map = manifest["qwen_kernel_source_map"]
    rmsnorm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_rmsnorm_input"
    )
    final_norm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_final_norm"
    )
    qk_norm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_attention_qk_norm"
    )
    attention_o = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_attention_o"
    )
    post_attention_norm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_rmsnorm_post_attention"
    )
    mlp_down = next(
        item for item in manifest["task_bodies"] if item["callable"] == "qwen_mlp_down"
    )

    assert rmsnorm["threading"] == "block"
    assert qk_norm["threading"] == "block"
    assert attention_o["threading"] == "block"
    assert post_attention_norm["threading"] == "block"
    assert post_attention_norm["consumes_fields"] == [
        "a",
        "b",
        "out",
        "tensor_args",
    ]
    assert "attention_residual" in post_attention_norm["consumes_roles"]
    assert final_norm["threading"] == "block"
    assert "for (unsigned int k = 0U; k < task->cols; ++k)" in source
    assert "task->scalar_args[0] == 1.0f" in full_source
    assert "task->scalar_args[1] == 0.0f" in full_source
    assert "rsqrtf(mean_square / static_cast<float>(task->cols) + 0.000001f)" in (
        full_source
    )
    assert "partial[0] / static_cast<float>(task->n)" not in full_source
    assert "const unsigned int row = static_cast<unsigned int>(j / task->cols);" in (
        full_source
    )
    assert "const unsigned int col = static_cast<unsigned int>(j % task->cols);" in (
        full_source
    )
    assert "static_cast<unsigned long long>(row) * input_stride;" in full_source
    assert "const unsigned int prompt_stride =" in full_source
    assert "const unsigned int requested_token_position =" in full_source
    assert "static_cast<unsigned long long>(token_row) * prompt_stride +" in (
        full_source
    )
    assert "pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f)" in full_source
    assert "qwen_rowwise_rmsnorm_batch_source" in manifest["implemented_contracts"]
    assert "qwen_final_norm" in full_source
    assert "qwen_attention_qk_norm" in full_source
    assert "for (unsigned long long j = threadIdx.x;" in full_source
    assert "const unsigned int col = static_cast<unsigned int>(j % task->cols);" in (
        full_source
    )
    assert "const unsigned int raw_q_width =" in full_source
    assert "raw_q_width >= head_dim ? raw_q_width / head_dim : 1U;" in full_source
    assert "const unsigned int q_width = query_heads * head_dim;" in full_source
    assert "const unsigned int kv_width = kv_heads * head_dim;" in full_source
    assert "const unsigned int qk_norm_input_stride =" in full_source
    assert "task->a_batch_stride > 0U ? task->a_batch_stride : task->cols;" in (
        full_source
    )
    assert "const bool is_query_region = col < q_width;" in full_source
    assert "const unsigned int source_col = is_query_region ?" in full_source
    assert "const unsigned int norm_slot = is_query_region ? 0U : 1U;" in (
        full_source
    )
    assert "const unsigned int kv_page_size =" in full_source
    assert "const unsigned int decode_position = task->scalar_arg_count > 2U" in (
        full_source
    )
    assert "static_cast<unsigned int>(task->scalar_args[2]) : 0U;" in full_source
    assert "const unsigned long long qk_norm_kv_write_index =" in full_source
    assert "task->c[qk_norm_kv_write_index] = task->out[j];" in full_source
    assert "const unsigned int *qk_norm_kv_page_table =" in full_source
    assert "task->tensor_arg_count > 4U && task->tensor_args[4]" in full_source
    assert (
        "reinterpret_cast<const unsigned int *>(task->tensor_args[4])"
        in full_source
    )
    assert "const unsigned int physical_page = qk_norm_kv_page_table ?" in (
        full_source
    )
    assert "qk_norm_kv_page_table[logical_page] : logical_page" in full_source
    assert (
        "const unsigned int row = static_cast<unsigned int>(j / task->cols);"
        in full_source
    )
    assert "qwen_qk_norm_batch_row_index_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_qkv_input_stride_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_normalized_k_cache_writeback_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_paged_k_cache_writeback_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_separate_qk_regions_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_block_rmsnorm_rope_source" in manifest[
        "implemented_contracts"
    ]
    assert "i % task->cols" in full_source
    assert "pto_cuda_tensor_arg_f32(task, 0U, col, 1.0f)" in full_source
    assert "qwen_input_rmsnorm_hidden_weight_source" in manifest[
        "implemented_contracts"
    ]
    assert "const unsigned int embedding_stride =" in full_source
    assert "const unsigned long long embedding_weight_index =" in full_source
    assert "static_cast<unsigned long long>(token_id) * embedding_stride" in (
        full_source
    )
    assert "qwen_embedding_shape_lookup_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_rmsnorm_post_attention" in full_source
    assert "const float residual_value = task->b ? task->b[j] : 0.0f;" in (
        full_source
    )
    assert "const float value = task->a[j] + residual_value;" in full_source
    assert "qwen_post_attention_residual_rmsnorm_source" in manifest[
        "implemented_contracts"
    ]
    assert mlp_down["consumes_fields"] == ["a", "b", "out", "tensor_args"]
    assert "mlp_residual" in mlp_down["consumes_roles"]
    assert (
        "const float residual_attention_value = task->b ? task->b[i] : 0.0f;"
        in full_source
    )
    assert "task->tensor_arg_count > 1U && task->tensor_args[1]" in full_source
    assert (
        "const float residual_value = pto_cuda_round_to_bf16_f32( "
        "residual_attention_value + residual_input_value);"
    ) in compact_source
    assert (
        "task->out[i] = pto_cuda_round_to_bf16_f32("
        "projected_down + residual_value);"
    ) in full_source
    assert "qwen_mlp_down_residual_add_source" in manifest["implemented_contracts"]
    assert (
        "qwen_post_attention_norm_full_rmsnorm_source"
        in manifest["implemented_contracts"]
    )
    assert "task->out[j] = task->a[j] * external_scale" in full_source
    assert "task->out[i] = task->a[i] * external_scale * norm_weight;" in (
        full_source
    )
    assert "for (unsigned long long j = threadIdx.x;" in source
    assert "external_scale * norm_weight" in source
    assert "pto_cuda_linear_arg_f32" in full_source
    assert "task->cols > 0U && task->inner > 0U" in full_source
    assert "const unsigned int kv_page_size =" in full_source
    assert "const unsigned int decode_position = task->scalar_arg_count > 2U" in (
        full_source
    )
    assert "const unsigned int sequence_capacity =" in full_source
    assert "const unsigned int cache_batch_size =" in full_source
    assert "const unsigned int kv_layer_index = task->scalar_arg_count > 3U" in (
        full_source
    )
    assert "const unsigned long long kv_layer_base =" in full_source
    assert "const unsigned int logical_page = decode_position / kv_page_size;" in (
        full_source
    )
    assert "const unsigned int page_offset = decode_position % kv_page_size;" in (
        full_source
    )
    assert "const unsigned int physical_page = kv_page_table ?" in full_source
    assert "const unsigned long long token_slot =" in full_source
    assert "const unsigned long long kv_write_index =" in full_source
    assert "kv_layer_base +" in full_source
    assert "static_cast<unsigned long long>(row) * sequence_capacity * kv_width" in (
        full_source
    )
    assert "task->c[kv_write_index] = projected;" in full_source
    assert "task->d[kv_write_index] = projected;" in full_source
    assert "partial[0] / static_cast<float>(task->inner)" in full_source
    assert "const float normalized = task->a[row_base + col] * scale * 0.5f" in (
        full_source
    )
    assert "(q_weight + k_weight)" in full_source
    assert "const unsigned int half_head_dim = head_dim >> 1U;" in full_source
    assert "const unsigned int half_inner = task->inner >> 1U;" in full_source
    assert "head_col + half_head_dim" in full_source
    assert "head_col - half_head_dim" in full_source
    assert "col + half_inner" in full_source
    assert "col - half_inner" in full_source
    assert "const unsigned int rope_index = first_half ?" in full_source
    assert "normalized * cos_value - paired * sin_value" in full_source
    assert "normalized * cos_value + paired * sin_value" in full_source
    assert "const unsigned int kv_window = task->inner;" in full_source
    assert "const unsigned int input_stride =" in full_source
    assert "task->a_batch_stride > 0U ? task->a_batch_stride : task->cols;" in (
        full_source
    )
    assert "const unsigned int query_head = col / head_dim;" in full_source
    assert "const unsigned int mapped_kv_head = query_head / heads_per_kv;" in (
        full_source
    )
    assert "mapped_kv_head < kv_heads ? mapped_kv_head : kv_heads - 1U" in (
        full_source
    )
    assert "const unsigned int *kv_page_table" in full_source
    assert "const unsigned int logical_page = step / kv_page_size;" in full_source
    assert "unsigned int attention_tile =" in full_source
    assert "for (unsigned int tile_begin = 0U; tile_begin < kv_window;" in (
        full_source
    )
    assert "const unsigned int tile_end =" in full_source
    assert "for (unsigned int step = tile_begin; step < tile_end; ++step)" in (
        full_source
    )
    assert "static_cast<unsigned long long>(physical_page) * kv_page_size" in (
        full_source
    )
    assert "const float attention_scale = rsqrtf(static_cast<float>(head_dim));" in (
        full_source
    )
    assert "const unsigned int query_base = query_head * head_dim;" in full_source
    assert "const unsigned int projection_query_base =" in full_source
    assert "for (unsigned int dim = 0U; dim < head_dim; ++dim)" in full_source
    assert "score += task->a[row_base + query_base + dim] *" in full_source
    assert "row_base + projection_query_base + dim" in full_source
    assert "__shared__ float attention_values[4096];" in full_source
    assert (
        "attention_values[projection_col] = pto_cuda_round_to_bf16_f32( "
        "projection_normalizer > 0.0f ? projection_weighted_value / "
        "projection_normalizer : 0.0f);"
    ) in compact_source
    assert "projected_attention +=" in full_source
    assert "attention_values[projection_col] * o_weight;" in full_source
    assert "score *= attention_scale;" in full_source
    assert "const unsigned long long value_kv_index =" in full_source
    assert "expf(score - max_score)" in full_source
    assert "expf(score - projection_max_score)" in full_source
    assert (
        "task->out[j] = pto_cuda_round_to_bf16_f32( "
        "normalizer > 0.0f ? weighted_value / normalizer : 0.0f);"
    ) in compact_source
    assert "const unsigned int projection_input_count =" in full_source
    assert "pto_cuda_tensor_arg_f32(" in full_source
    assert "task, 0U, o_weight_index, 0.0f" in full_source
    assert "const unsigned int sequence_capacity =" in full_source
    assert "const unsigned long long kv_read_base =" in full_source
    assert full_source.count("kv_layer_base +") >= 4
    assert full_source.count("kv_read_base +") == 6
    assert "qwen_attention_o_batch_local_kv_read_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_attention_o_qk_norm_input_stride_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_attention_o_bounded_projection_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_attention_o_cached_projection_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_decode_attention_head_dim_scale_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_decode_attention_dot_product_source" in manifest[
        "implemented_contracts"
    ]
    assert (
        "task->out[i] = pto_cuda_round_to_bf16_f32( "
        "pto_cuda_silu(gate_value) * up_value);"
    ) in compact_source
    assert "task->out[j] = pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f)" in (
        full_source
    )
    assert "const unsigned int logits_tile =" in full_source
    assert "requested_logits_tile > 0U ? requested_logits_tile : 256U" in (
        full_source
    )
    assert "const unsigned int active_logits_cols =" in full_source
    assert "task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1)" in (
        full_source
    )
    assert "i < active_logits_elements" in full_source
    assert "for (unsigned int tile_begin = 0U; tile_begin < hidden_width;" in (
        full_source
    )
    assert "acc += task->a[a_index] *" in full_source
    assert "pto_cuda_tensor_arg_f32(task, 0U, weight_index, 0.0f)" in full_source
    assert "for (unsigned int token = 1; token < task->cols; ++token)" not in (
        full_source
    )
    assert "sampled_tokens" not in full_source
    logits = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_logits"
    )
    assert logits["threading"] == "block"
    assert "__shared__ float logits_best_values[1024];" in full_source
    assert "__shared__ unsigned int logits_best_tokens[1024];" in full_source
    assert "for (unsigned int token = threadIdx.x; token < active_logits_cols;" in (
        full_source
    )
    assert "if (candidate > local_best_logit)" in full_source
    assert "active_projection_cols" in full_source
    assert "col < active_projection_cols" in full_source
    assert "task->scalar1 > 0.0f ? static_cast<unsigned int>(task->scalar1)" in (
        full_source
    )
    assert "logits_best_values[threadIdx.x] = local_best_logit;" in full_source
    assert "logits_best_values[threadIdx.x + stride] >" in full_source
    assert "output_ids[decode_step] = logits_best_tokens[0];" in full_source
    assert "output_ids[decode_step] = best_token;" in full_source
    assert "const unsigned long long next_input_index =" in full_source
    assert "static_cast<unsigned long long>(task->scalar_args[2]) + 1ULL;" in (
        full_source
    )
    assert "input_ids[feedback_input_index] = logits_best_tokens[0];" in full_source
    assert "input_ids[feedback_input_index] = best_token;" in full_source
    assert "input_ids[0] = best_token;" not in full_source
    assert "qwen_unit_math_source_coverage" in manifest["implemented_contracts"]
    assert "qwen_shape_field_linear_projection_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_shape_field_qk_rmsnorm_source" in manifest["implemented_contracts"]
    assert "qwen_final_norm_full_rmsnorm_source" in manifest["implemented_contracts"]
    assert "qwen_shape_field_qk_rope_source" in manifest["implemented_contracts"]
    assert "qwen_qk_norm_rotate_half_rope_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_bounded_decode_attention_reduction_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_decode_attention_dot_product_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_gqa_decode_attention_head_grouping_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_paged_kv_attention_index_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_layer_partitioned_kv_cache_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_slot_mapped_kv_cache_writeback_source" in manifest[
        "implemented_contracts"
    ]
    assert manifest["coverage"]["kv_write_policy"] == (
        "slot_mapped_kv_cache_writeback_ready"
    )
    kv_entry = next(
        item
        for item in source_map["entries"]
        if item["pto_callables"] == ["qwen_attention_qkv"]
    )
    assert kv_entry["pto_status"] == "slot_mapped_kv_cache_writeback_source_ready"
    assert "qwen_tiled_decode_attention_softmax_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_logits_full_vocab_argmax_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_logits_tiled_vocab_projection_source" in manifest[
        "implemented_contracts"
    ]
    logits_entry = next(
        item
        for item in source_map["entries"]
        if item["pto_callables"] == ["qwen_logits"]
    )
    assert logits_entry["pto_status"] == "tiled_vocab_projection_source_ready"
    assert "qwen_logits_device_sampled_token_feedback_source" in manifest[
        "implemented_contracts"
    ]


def test_qwen_task_bodies_do_not_exit_grid_stride_wrapper_early():
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import body_specs

    for spec in body_specs():
        assert "return;" not in spec["body"], spec["callable"]


def test_task_body_manifest_tracks_qwen_tensor_tile_source_contract():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    contract = manifest["qwen_tensor_tile_contract"]

    assert contract["status"] == "qwen_tensor_tile_source_contract_ready"
    assert contract["wmma"] == {
        "api": "nvcuda::wmma",
        "mma_shape": "m16n16k8",
        "input": "tf32",
        "accumulator": "f32",
    }
    assert [task["id"] for task in contract["task_functions"]] == [
        "qwen_attention_projection_tile",
        "qwen_mlp_projection_tile",
    ]
    assert [task["tensor_tile"] for task in contract["task_functions"]] == [
        {"rows": 16, "cols": 64, "inner": 128},
        {"rows": 16, "cols": 64, "inner": 256},
    ]
    assert contract["rendered_source"]["required_fragments"] == [
        "task->rows != 16U",
        "task->cols != 64U",
        "k < 128U",
        "k < 256U",
        "wmma::mma_sync",
    ]
    assert "qwen_tensor_tile_source_contract" in manifest["implemented_contracts"]
    assert "capture multi-repeat A100/H200 throughput rows" in contract[
        "remaining_wiring"
    ]


def test_task_body_manifest_tracks_external_kernel_source_map():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    source_map = manifest["qwen_kernel_source_map"]

    assert source_map["status"] == "qwen_kernel_source_map_ready"
    assert {item["project"] for item in source_map["reference_snapshots"]} == {
        "FlashInfer",
        "SGLang",
        "vLLM",
    }
    mapped_callables = {
        callable_name
        for entry in source_map["entries"]
        for callable_name in entry["pto_callables"]
    }
    assert {
        "qwen_attention_qkv",
        "qwen_attention_qk_norm",
        "qwen_logits",
        "qwen_mlp_gate_up",
        "qwen_rmsnorm_input",
    } <= mapped_callables
    assert any(
        reference["path"] == "csrc/libtorch_stable/activation_kernels.cu"
        for entry in source_map["entries"]
        for reference in entry["reference_files"]
    )
    assert "qwen_kernel_source_map" in manifest["implemented_contracts"]


def test_task_body_manifest_tracks_qwen_decode_attention_oracle():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    oracle = manifest["qwen_decode_attention_oracle"]

    assert oracle["status"] == "qwen_decode_attention_oracle_ready"
    assert oracle["scope"] == "bounded_two_step_gqa_hidden8_reference"
    assert oracle["head_grouping"] == {
        "query_heads": 4,
        "kv_heads": 2,
        "head_dim": 2,
        "heads_per_kv": 2,
    }
    assert oracle["steps"]["attention_context"] == [
        5.659033,
        11.318066,
        5.5,
        11.0,
        15.689571,
        20.919428,
        18.162065,
        24.216087,
    ]
    assert oracle["steps"]["attention_probability_by_col"] == [
        [0.48233, 0.51767],
        [0.48233, 0.51767],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.530016, 0.469984],
        [0.530016, 0.469984],
        [0.438442, 0.561558],
        [0.438442, 0.561558],
    ]
    assert "dot(query_head, key_cache[step][kv_head])" in oracle["equation"]
    assert "qwen_bounded_decode_attention_reduction_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_gqa_decode_attention_head_grouping_source" in manifest[
        "implemented_contracts"
    ]
