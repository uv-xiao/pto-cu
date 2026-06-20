import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_import_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_import_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_source_fixture(root):
    files = {
        "vllm/models/deepseek_v4/nvidia/model.py": "class DeepseekV4ForCausalLM:\n    pass\n",
        "vllm/tokenizers/deepseek_v4.py": "class DeepseekV4Tokenizer:\n    pass\n",
        "vllm/transformers_utils/configs/deepseek_v4.py": (
            "class DeepseekV4Config:\n    pass\n"
        ),
        "vllm/models/deepseek_v4/quant_config.py": (
            "class DeepseekV4FP8Config:\n    pass\n"
            "def deepseek_v4_fp8():\n    pass\n"
        ),
    }
    for relative_path, text in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_probe_records_source_contract_from_fixture_when_vllm_is_missing(
    monkeypatch, tmp_path
):
    probe = load_probe_module()
    write_source_fixture(tmp_path)

    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else importlib.util.find_spec(name),
    )

    result = probe.run_probe(source_root=tmp_path)

    assert result["status"] == "skipped"
    assert result["vllm_import"] == "missing"
    assert result["source_status"] == "available"
    assert result["source_symbols"] == [
        "DeepseekV4Config",
        "DeepseekV4FP8Config",
        "DeepseekV4ForCausalLM",
        "DeepseekV4Tokenizer",
    ]


def test_probe_require_vllm_fails_when_import_is_missing(monkeypatch, tmp_path, capsys):
    probe = load_probe_module()
    write_source_fixture(tmp_path)
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else importlib.util.find_spec(name),
    )

    assert probe.main(["--source-root", str(tmp_path), "--require-vllm"]) == 2
    assert '"vllm_import": "missing"' in capsys.readouterr().out


def test_probe_imports_deepseek_symbols_when_vllm_is_available(monkeypatch, tmp_path):
    probe = load_probe_module()
    write_source_fixture(tmp_path)
    fake_modules = {}

    def install_module(module_name, attr_name):
        module = types.ModuleType(module_name)
        setattr(module, attr_name, type(attr_name, (), {}))
        fake_modules[module_name] = module
        monkeypatch.setitem(sys.modules, module_name, module)

    install_module("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")
    install_module("vllm.tokenizers.deepseek_v4", "DeepseekV4Tokenizer")
    install_module(
        "vllm.transformers_utils.configs.deepseek_v4", "DeepseekV4Config"
    )
    install_module("vllm.models.deepseek_v4.quant_config", "DeepseekV4FP8Config")

    def fake_find_spec(name):
        if name == "vllm":
            return importlib.machinery.ModuleSpec("vllm", loader=None)
        return importlib.util.find_spec(name)

    monkeypatch.setattr(probe.importlib.util, "find_spec", fake_find_spec)

    result = probe.run_probe(source_root=tmp_path)

    assert result["status"] == "passed"
    assert result["vllm_import"] == "available"
    assert result["imported_symbols"] == [
        "DeepseekV4Config",
        "DeepseekV4FP8Config",
        "DeepseekV4ForCausalLM",
        "DeepseekV4Tokenizer",
    ]
    assert result["import_errors"] == []
