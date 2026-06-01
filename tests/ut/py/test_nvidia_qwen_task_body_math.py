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
    assert "cuda_task_bodies_match_qwen_unit_math_oracle" in manifest[
        "remaining_runtime_gaps"
    ]
