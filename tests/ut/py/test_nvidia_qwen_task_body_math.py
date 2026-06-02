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


def test_generated_source_contains_qwen_unit_math_kernels():
    module = load_task_bodies_module()
    sys.path.insert(0, str(ROOT / "examples" / "cuda"))
    from qwen_persistent_task_bodies_impl.lifecycle import task_functions
    from simpler_setup.cuda_callable_compiler import render_persistent_dag_source

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    source = manifest["rendered_source"]["preview"]
    full_source = render_persistent_dag_source(task_functions())
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
    post_attention_norm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_rmsnorm_post_attention"
    )

    assert "rsqrtf(partial[0] / static_cast<float>(task->n) + 0.000001f)" in source
    assert rmsnorm["threading"] == "block"
    assert qk_norm["threading"] == "block"
    assert post_attention_norm["threading"] == "block"
    assert final_norm["threading"] == "block"
    assert "__shared__ float partial[1024];" in source
    assert (
        "rsqrtf(partial[0] / static_cast<float>(task->n) + 0.000001f)"
        in full_source
    )
    assert "qwen_final_norm" in full_source
    assert "qwen_attention_qk_norm" in full_source
    assert "for (unsigned long long j = threadIdx.x;" in full_source
    assert "const unsigned int col = static_cast<unsigned int>(j % task->cols);" in (
        full_source
    )
    assert "const unsigned int q_width = query_heads * head_dim;" in full_source
    assert "const unsigned int kv_width = kv_heads * head_dim;" in full_source
    assert "const bool is_query_region = col < q_width;" in full_source
    assert "const unsigned int source_col = is_query_region ?" in full_source
    assert "const unsigned int norm_slot = is_query_region ? 0U : 1U;" in (
        full_source
    )
    assert "const unsigned int kv_page_size =" in full_source
    assert "const unsigned int decode_position = task->scalar_arg_count > 2U" in (
        full_source
    )
    assert "const unsigned long long qk_norm_kv_write_index =" in full_source
    assert "task->c[qk_norm_kv_write_index] = task->out[j];" in full_source
    assert "qwen_qk_norm_normalized_k_cache_writeback_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_separate_qk_regions_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_qk_norm_block_rmsnorm_rope_source" in manifest[
        "implemented_contracts"
    ]
    assert "const unsigned int hidden_col =" in full_source
    assert "const unsigned int embedding_stride =" in full_source
    assert "const unsigned long long embedding_weight_index =" in full_source
    assert "static_cast<unsigned long long>(token_id) * embedding_stride" in (
        full_source
    )
    assert "qwen_embedding_shape_lookup_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_rmsnorm_post_attention" in full_source
    assert (
        "qwen_post_attention_norm_full_rmsnorm_source"
        in manifest["implemented_contracts"]
    )
    assert "task->out[j] = task->a[j] * external_scale" in full_source
    assert "task->out[i] = task->a[i] * external_scale" not in full_source
    assert "for (unsigned long long j = threadIdx.x;" in source
    assert "task->scalar_args[1] * norm_weight" in source
    assert "pto_cuda_linear_arg_f32" in full_source
    assert "task->cols > 0U && task->inner > 0U" in full_source
    assert "const unsigned int kv_page_size =" in full_source
    assert "const unsigned int decode_position = task->scalar_arg_count > 2U" in (
        full_source
    )
    assert "const unsigned int sequence_capacity =" in full_source
    assert "const unsigned int logical_page = decode_position / kv_page_size;" in (
        full_source
    )
    assert "const unsigned int page_offset = decode_position % kv_page_size;" in (
        full_source
    )
    assert "const unsigned int physical_page = kv_page_table ?" in full_source
    assert "const unsigned long long token_slot =" in full_source
    assert "const unsigned long long kv_write_index =" in full_source
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
    assert "const unsigned int pair_col = col ^ 1U;" in full_source
    assert "const unsigned int rope_index = col >> 1U;" in full_source
    assert "normalized * cos_value - paired * sin_value" in full_source
    assert "normalized * cos_value + paired * sin_value" in full_source
    assert "const unsigned int kv_window = task->inner;" in full_source
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
    assert "expf(query * task->c[kv_index] - max_score)" in full_source
    assert "weighted_value / normalizer" in full_source
    assert "const unsigned int projection_input_count =" in full_source
    assert "pto_cuda_tensor_arg_f32(task, 0U, o_weight_index, 0.0f)" in (
        full_source
    )
    assert "projected_attention += attention_value * o_weight;" in full_source
    assert "qwen_attention_o_bounded_projection_source" in manifest[
        "implemented_contracts"
    ]
    assert "task->out[i] = pto_cuda_silu(gate_value) * up_value;" in full_source
    assert "task->out[i] = pto_cuda_linear_arg_f32(task, 0U, row, col, 0.0f)" in (
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
    assert "input_ids[0] = best_token;" in full_source
    assert "qwen_unit_math_source_coverage" in manifest["implemented_contracts"]
    assert "qwen_shape_field_linear_projection_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_shape_field_qk_rmsnorm_source" in manifest["implemented_contracts"]
    assert "qwen_final_norm_full_rmsnorm_source" in manifest["implemented_contracts"]
    assert "qwen_shape_field_qk_rope_source" in manifest["implemented_contracts"]
    assert "qwen_bounded_decode_attention_reduction_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_gqa_decode_attention_head_grouping_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_paged_kv_attention_index_source" in manifest[
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
        5.63496,
        11.179976,
        5.769676,
        10.460647,
        16.16257,
        20.921294,
        16.432501,
        25.205456,
    ]
    assert oracle["steps"]["attention_probability_by_col"] == [
        [0.485004, 0.514996],
        [0.490001, 0.509999],
        [0.470036, 0.529964],
        [0.529964, 0.470036],
        [0.512497, 0.487503],
        [0.529964, 0.470036],
        [0.5025, 0.4975],
        [0.41096, 0.58904],
    ]
    assert "qwen_bounded_decode_attention_reduction_source" in manifest[
        "implemented_contracts"
    ]
    assert "qwen_gqa_decode_attention_head_grouping_source" in manifest[
        "implemented_contracts"
    ]
