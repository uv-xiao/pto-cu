import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_config_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_config_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_source_fixture(root):
    files = {
        "vllm/transformers_utils/configs/deepseek_v4.py": (
            "class DeepseekV4Config:\n"
            "    model_type = \"deepseek_v4\"\n"
            "    max_position_embeddings: int = 1048576\n"
        ),
        "vllm/models/deepseek_v4/quant_config.py": (
            "class DeepseekV4FP8Config:\n"
            "    def get_name(cls):\n"
            "        return \"deepseek_v4_fp8\"\n"
            "expert_dtype = \"fp4\"\n"
        ),
    }
    for relative_path, text in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_config_probe_records_source_contract_when_vllm_is_missing(
    monkeypatch, tmp_path
):
    probe = load_probe_module()
    write_source_fixture(tmp_path)
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else importlib.util.find_spec(name),
    )

    result = probe.run_probe(source_root=tmp_path, max_position_embeddings=262144)

    assert result["status"] == "skipped"
    assert result["vllm_import"] == "missing"
    assert result["source_status"] == "available"
    assert result["source_contract"] == {
        "config_class": "DeepseekV4Config",
        "default_max_position_embeddings": 1048576,
        "requested_max_position_embeddings": 262144,
        "quantization_method": "deepseek_v4_fp8",
    }


def test_config_probe_require_vllm_fails_when_package_is_missing(
    monkeypatch, tmp_path, capsys
):
    probe = load_probe_module()
    write_source_fixture(tmp_path)
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else importlib.util.find_spec(name),
    )

    assert probe.main(["--source-root", str(tmp_path), "--require-vllm"]) == 2
    payload = capsys.readouterr().out
    assert '"status": "skipped"' in payload
    assert '"vllm_import": "missing"' in payload


def test_config_probe_builds_synthetic_config_when_vllm_is_available(
    monkeypatch, tmp_path
):
    probe = load_probe_module()
    write_source_fixture(tmp_path)

    config_module = types.ModuleType("vllm.transformers_utils.configs.deepseek_v4")
    quant_module = types.ModuleType("vllm.models.deepseek_v4.quant_config")

    class DeepseekV4Config:
        model_type = "deepseek_v4"

        def __init__(self, max_position_embeddings, expert_dtype, quantization_config):
            self.max_position_embeddings = max_position_embeddings
            self.expert_dtype = expert_dtype
            self.quantization_config = quantization_config

    class DeepseekV4FP8Config:
        @classmethod
        def get_name(cls):
            return "deepseek_v4_fp8"

        @classmethod
        def override_quantization_method(cls, quantization_config, user_quant, hf_config):
            assert hf_config.quantization_config == quantization_config
            assert user_quant is None
            return "deepseek_v4_fp8"

    config_module.DeepseekV4Config = DeepseekV4Config
    quant_module.DeepseekV4FP8Config = DeepseekV4FP8Config
    monkeypatch.setitem(
        sys.modules, "vllm.transformers_utils.configs.deepseek_v4", config_module
    )
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.quant_config", quant_module)

    def fake_find_spec(name):
        if name == "vllm":
            return importlib.machinery.ModuleSpec("vllm", loader=None)
        return importlib.util.find_spec(name)

    monkeypatch.setattr(probe.importlib.util, "find_spec", fake_find_spec)

    result = probe.run_probe(source_root=tmp_path, max_position_embeddings=262144)

    assert result["status"] == "passed"
    assert result["vllm_import"] == "available"
    assert result["config_probe"] == {
        "config_class": "DeepseekV4Config",
        "model_type": "deepseek_v4",
        "max_position_embeddings": 262144,
        "expert_dtype": "fp4",
        "quantization_method": "deepseek_v4_fp8",
        "override_quantization_method": "deepseek_v4_fp8",
    }
