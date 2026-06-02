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


def test_generated_source_contains_qwen_unit_math_kernels():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    source = manifest["rendered_source"]["preview"]
    rmsnorm = next(
        item
        for item in manifest["task_bodies"]
        if item["callable"] == "qwen_rmsnorm_input"
    )

    assert "rsqrtf(partial[0] / static_cast<float>(task->n) + 0.000001f)" in source
    assert rmsnorm["threading"] == "block"
    assert "__shared__ float partial[1024];" in source
    assert "for (unsigned long long j = threadIdx.x;" in source
    assert "task->scalar_args[1] * norm_weight" in source
    assert "task->c[kv_index] = k;" in source
    assert "task->d[kv_index] = v;" in source
    assert "task->out[i] = v;" in source
    assert "1.0f + expf(-gate_value)" in source
    assert "task->out[i] = silu_gate * up_value;" in source
    assert "lm_head" in source
    assert "output_ids[decode_step] = best_token;" in source
    assert "input_ids[0] = best_token;" in source
    assert "qwen_unit_math_source_coverage" in manifest["implemented_contracts"]
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
