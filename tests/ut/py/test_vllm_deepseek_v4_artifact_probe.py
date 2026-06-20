import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_artifact_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_artifact_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_artifact_fixture(root, *, missing_second_shard=False):
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "max_position_embeddings": 1048576,
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "DeepseekV4Tokenizer",
                "model_max_length": 1048576,
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "chat_template": "{{ messages }}",
            }
        ),
        encoding="utf-8",
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    if not missing_second_shard:
        (root / "model-00002-of-00002.safetensors").write_bytes(b"1234567")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )


def test_missing_artifact_directory_skips_by_default(monkeypatch, tmp_path):
    probe = load_probe_module()
    monkeypatch.setattr(probe, "_run_import_probe", lambda source_root: None)
    monkeypatch.setattr(
        probe,
        "_run_config_probe",
        lambda source_root, max_position_embeddings: {
            "status": "skipped",
            "vllm_import": "missing",
        },
    )

    result = probe.run_probe(artifact_dir=tmp_path / "missing")

    assert result["status"] == "skipped"
    assert result["artifact_probe"]["status"] == "skipped"
    assert result["artifact_probe"]["reason"] == "artifact directory is missing"
    assert result["vllm_status"] == "missing"
    assert result["non_claim"] == "not model-load or serving evidence"
    assert "can_serve" not in result


def test_require_artifacts_fails_when_directory_is_missing(tmp_path, capsys):
    probe = load_probe_module()

    code = probe.main(
        [
            "--artifact-dir",
            str(tmp_path / "missing"),
            "--require-artifacts",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["artifact_probe"]["status"] == "failed"
    assert payload["artifact_probe"]["reason"] == "artifact directory is missing"


def test_complete_artifacts_report_config_tokenizer_and_shard_fields(
    monkeypatch, tmp_path
):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_artifact_fixture(artifact_dir)
    monkeypatch.setattr(probe, "_run_import_probe", lambda source_root: None)
    monkeypatch.setattr(
        probe,
        "_run_config_probe",
        lambda source_root, max_position_embeddings: {
            "status": "skipped",
            "vllm_import": "missing",
        },
    )

    result = probe.run_probe(artifact_dir=artifact_dir)

    artifact = result["artifact_probe"]
    assert result["status"] == "skipped"
    assert artifact["status"] == "passed"
    assert artifact["required_files"]["missing"] == []
    assert artifact["tokenizer_files_present"] == ["tokenizer.json"]
    assert artifact["config_fields"] == {
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 7168,
        "max_position_embeddings": 1048576,
        "model_type": "deepseek_v4",
        "num_attention_heads": 128,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "quantization_config": {"quant_method": "fp8"},
        "torch_dtype": "bfloat16",
    }
    assert artifact["tokenizer_config_fields"] == {
        "bos_token": "<bos>",
        "chat_template_present": True,
        "eos_token": "<eos>",
        "model_max_length": 1048576,
        "tokenizer_class": "DeepseekV4Tokenizer",
    }
    assert artifact["indexed_tensors"] == 2
    assert artifact["indexed_shards"] == 2
    assert artifact["present_shards"] == 2
    assert artifact["missing_shards"] == 0
    assert artifact["present_bytes"] == 12
    assert artifact["index_total_size"] == 12


def test_missing_shards_fail_only_when_artifacts_are_required(tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_artifact_fixture(artifact_dir, missing_second_shard=True)

    skipped = probe.run_probe(artifact_dir=artifact_dir)
    failed = probe.run_probe(artifact_dir=artifact_dir, require_artifacts=True)

    assert skipped["status"] == "skipped"
    assert skipped["artifact_probe"]["status"] == "skipped"
    assert skipped["artifact_probe"]["missing_shards"] == 1
    assert failed["status"] == "failed"
    assert failed["artifact_probe"]["status"] == "failed"
    assert failed["artifact_probe"]["missing_examples"] == [
        "model-00002-of-00002.safetensors"
    ]


def test_require_vllm_fails_when_vllm_is_missing(monkeypatch, tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_artifact_fixture(artifact_dir)
    monkeypatch.setattr(probe, "_run_import_probe", lambda source_root: None)
    monkeypatch.setattr(
        probe,
        "_run_config_probe",
        lambda source_root, max_position_embeddings: {
            "status": "skipped",
            "vllm_import": "missing",
        },
    )

    result = probe.run_probe(artifact_dir=artifact_dir, require_vllm=True)

    assert result["status"] == "failed"
    assert result["artifact_probe"]["status"] == "passed"
    assert result["vllm_status"] == "missing"
    assert result["failure_reasons"] == ["vLLM is required but not available"]


def test_complete_artifacts_and_vllm_probes_pass(monkeypatch, tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_artifact_fixture(artifact_dir)
    monkeypatch.setattr(
        probe,
        "_run_import_probe",
        lambda source_root: {
            "status": "passed",
            "vllm_import": "available",
            "import_errors": [],
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_config_probe",
        lambda source_root, max_position_embeddings: {
            "status": "passed",
            "vllm_import": "available",
            "config_probe": {"model_type": "deepseek_v4"},
        },
    )

    result = probe.run_probe(artifact_dir=artifact_dir)

    assert result["status"] == "passed"
    assert result["artifact_probe"]["status"] == "passed"
    assert result["vllm_status"] == "available"
    assert result["vllm_import_probe"]["status"] == "passed"
    assert result["vllm_config_probe"]["status"] == "passed"
