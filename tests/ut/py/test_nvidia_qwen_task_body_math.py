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
    assert "qwen_unit_math_oracle" in manifest["implemented_contracts"]
    assert "cuda_live_qwen_unit_math_execution" in manifest[
        "remaining_runtime_gaps"
    ]


def test_generated_source_contains_qwen_unit_math_kernels():
    module = load_task_bodies_module()

    manifest = module.build_task_body_manifest(num_hidden_layers=1)
    source = manifest["rendered_source"]["preview"]

    assert "rsqrtf(mean_square + 0.000001f)" in source
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
